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

### This is a custom model class that adds dropout layers to the DeBERTa model
class DebertaWithDropout(torch.nn.Module):
    def __init__(self, model_name='microsoft/deberta-v3-base', num_labels=2,
                 hidden_dropout_rate=0.1, attention_dropout_rate=0.1, classifier_dropout_rate=0.2):
        super().__init__()
        self.num_labels = num_labels
        
        # Load base model configuration
        config = DebertaV2Config.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
        # Override dropout rates in the config for the base model
        config.hidden_dropout_prob = hidden_dropout_rate
        config.attention_probs_dropout_prob = attention_dropout_rate
        
        # Load the base DeBERTa model (without the classification head initially)
        self.deberta = DebertaV2Model.from_pretrained(model_name, config=config)
        
        # Define the dropout layer to be applied before the classifier
        self.classifier_dropout = torch.nn.Dropout(classifier_dropout_rate)
        
        # Define the classifier layer
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        
        # Store config for potential later use
        self.config = config
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, inputs_embeds=None, labels=None,
                output_attentions=None, output_hidden_states=None, return_dict=None):
                
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get outputs from the base DeBERTa model
        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        
        # Standard pooling for sequence classification: Use the hidden state of the first token
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0]
        
        # Apply classifier dropout only during training
        if self.training:
            pooled_output = self.classifier_dropout(pooled_output)
            
        # Pass the pooled output through the classifier to get logits
        logits = self.classifier(pooled_output)
        
        loss = None
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
            
        # Return results in the standard HF SequenceClassifierOutput format
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

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
        MODEL_PATH = 'DetectorTemplate/DetectorCode/best_model_BCE.pt'                      
        BATCH_SIZE = 16                                   
        HIDDEN_DROPOUT_RATE = 0.1
        ATTENTION_DROPOUT_RATE = 0.1
        CLASSIFIER_DROPOUT_RATE = 0.2  
        
        unlabeled_data = unlabeled_processed_data
        unlabeled_dataset = TensorDataset(
            unlabeled_data['input_ids'],
            unlabeled_data['attention_mask'],
        )


        unlabeled_data_loader = DataLoader(unlabeled_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Load the Trained Model
        model = DebertaWithDropout(
            hidden_dropout_rate=HIDDEN_DROPOUT_RATE,
            attention_dropout_rate=ATTENTION_DROPOUT_RATE,
            classifier_dropout_rate=CLASSIFIER_DROPOUT_RATE
        )  # Initialize with the same dropout rates
        # Move model to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

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
