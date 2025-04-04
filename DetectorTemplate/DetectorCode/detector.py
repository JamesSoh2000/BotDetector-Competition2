from abc_classes import ADetector
from teams_classes import DetectionMark
from collections import Counter
from datetime import datetime
import re

import json
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
import sentencepiece

class Detector(ADetector):
    def detect_bot(self, session_data):
        marked_account = []
        
        processed_data = self._process_data(session_data)
        users_confidence = self._calculate_confidence(processed_data)

        for user_confidence in users_confidence:
            is_bot = user_confidence['predicted_label'] > 40
            marked_account.append(DetectionMark(
                user_id=user_confidence['user_id'],
                confidence=user_confidence['predicted_label'],
                bot=is_bot
            ))

        return marked_account

####################### 1. Data Processor#######################
    def _process_data(self, unlabeled_data):

       
        users_dict = {}
        for user in unlabeled_data.users:
            users_dict[user['id']] = {
                'description': user['description'],
                'posts': [],
                
            }

        
        for post in unlabeled_data.posts:
            author_id = post['author_id']
            if author_id in users_dict:
                users_dict[author_id]['posts'].append(post['text'])

        user_ids = list(users_dict.keys())
        

        def get_concatenated_texts(user_ids, users_dict):
            texts = []
            for user_id in user_ids:
                user = users_dict[user_id]
                description = user['description'] if user['description'] else ""
                posts_text = " [SEP] ".join(user['posts'])
                if description:
                    concatenated_text = description + " [SEP] " + posts_text
                else:
                    concatenated_text = posts_text
                texts.append(concatenated_text)
            return texts

        
        tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base')

        
        unlabeled_texts = get_concatenated_texts(user_ids, users_dict)
        unlabeled_tokenized = tokenizer(
            unlabeled_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        unlabeled_processed_data = {
            'input_ids': unlabeled_tokenized['input_ids'],
            'attention_mask': unlabeled_tokenized['attention_mask'],
            'user_ids': user_ids 
            
        }

        
        return unlabeled_processed_data



################## 2. Evaluate the data with the model #######################
    def _calculate_confidence(self, unlabeled_processed_data):
        MODEL_PATH = 'DetectorTemplate/DetectorCode/best_model_Fr.pt'                      
        BATCH_SIZE = 16                                   

        
        unlabeled_data = unlabeled_processed_data
        unlabeled_dataset = TensorDataset(
            unlabeled_data['input_ids'],
            unlabeled_data['attention_mask'],
        )


        unlabeled_data_loader = DataLoader(unlabeled_dataset, batch_size=BATCH_SIZE, shuffle=False)

       
        model = DebertaV2ForSequenceClassification.from_pretrained(
            'microsoft/deberta-v3-base',
            num_labels=2 
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)


        model.eval()  
        predictions = []  
        user_ids = unlabeled_data['user_ids'] 

        with torch.no_grad(): 
            for batch in unlabeled_data_loader:
                input_ids, attention_mask = [x.to(device) for x in batch]
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
                predicted_classes = torch.argmax(logits, dim=1).cpu().tolist() 
                predictions.extend(predicted_classes)


        output_data = []
        for user_id, prediction in zip(user_ids, predictions):
            predicted_label = 60 if prediction == 1 else 40
            output_data.append({
                "user_id": user_id,
                "predicted_label": predicted_label
            })
        return output_data
