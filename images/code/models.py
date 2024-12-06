
from transformers import AutoModelForImageClassification
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.modeling_outputs import ImageClassifierOutput
import torch.nn.functional as F
import torch
from utils import ImgsData

# @dataclass
# class ImageClassifierOutput(ModelOutput):

#     loss: Optional[torch.FloatTensor] = None
#     logits: torch.FloatTensor = None
#     hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
#     attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class My_last_hidden_state_output:
    img_last_hidden_state: any
    RSJ_last_hidden_state: any
    LSJ_last_hidden_state: any
    RHIP_last_hidden_state: any
    LHIP_last_hidden_state: any


# "google/vit-base-patch16-224"
class CustomVitForMultiLabelImageClassification(AutoModelForImageClassification):
    def forward(self, imgs_batch: ImgsData, labels=None, **kwargs):
        # Your custom forward logic here
        imgs_tensor = imgs_batch.img
        RSJ_imgs_tensor = imgs_batch.RSJ_imgs
        LSJ_imgs_tensor = imgs_batch.LSJ_imgs
        RHIP_imgs_tensor = imgs_batch.RHIP_imgs
        LHIP_imgs_tensor = imgs_batch.LHIP_imgs
        # Get last hidden state
        img_last_hidden_state = self.base_model(imgs_tensor)["last_hidden_state"]
        RSJ_last_hidden_state = self.base_model(RSJ_imgs_tensor)["last_hidden_state"]
        LSJ_last_hidden_state = self.base_model(LSJ_imgs_tensor)["last_hidden_state"]
        RHIP_last_hidden_state = self.base_model(RHIP_imgs_tensor)["last_hidden_state"]
        LHIP_last_hidden_state = self.base_model(LHIP_imgs_tensor)["last_hidden_state"]
        # Get logits
        img_logits = self.classifier(img_last_hidden_state[:, 0, :])
        RSJ_logits = self.classifier(RSJ_last_hidden_state[:, 0, :])
        LSJ_logits = self.classifier(LSJ_last_hidden_state[:, 0, :])
        RHIP_logits = self.classifier(RHIP_last_hidden_state[:, 0, :])
        LHIP_logits = self.classifier(LHIP_last_hidden_state[:, 0, :])

        
        all_logits = torch.stack(
            [
                img_logits,
                RSJ_logits,
                LSJ_logits,
                RHIP_logits,
                LHIP_logits,
            ],
            dim=0,
        )

        
        if self.method == "average":
            final_logits = torch.mean(all_logits, dim=0)

        
        elif self.method == "attention":
            attention_scores = F.softmax(self.attention_weights, dim=0)  
            attention_scores = attention_scores.view(5, 1, 1)  
            # print("all_logits shape:", all_logits.shape)
            # print("attention_scores shape:", attention_scores.shape)
            final_logits = torch.sum(all_logits * attention_scores, dim=0)  

        # Comput loss
        loss = None
        # loss function
        loss_fn = nn.BCEWithLogitsLoss()
        # Comput loss
        loss = loss_fn(final_logits, labels)
        # Custom logic can be added here
        return ImageClassifierOutput(
            loss=loss,
            logits=final_logits,
            hidden_states=None,
            attentions=None,
        )

    def predict(self, imgs_batch: ImgsData, labels=None, **kwargs):
        # Your custom forward logic here
        imgs_tensor = imgs_batch.img
        RSJ_imgs_tensor = imgs_batch.RSJ_imgs
        LSJ_imgs_tensor = imgs_batch.LSJ_imgs
        RHIP_imgs_tensor = imgs_batch.RHIP_imgs
        LHIP_imgs_tensor = imgs_batch.LHIP_imgs
        # Get last hidden state
        img_last_hidden_state = self.base_model(imgs_tensor)["last_hidden_state"]
        RSJ_last_hidden_state = self.base_model(RSJ_imgs_tensor)["last_hidden_state"]
        LSJ_last_hidden_state = self.base_model(LSJ_imgs_tensor)["last_hidden_state"]
        RHIP_last_hidden_state = self.base_model(RHIP_imgs_tensor)["last_hidden_state"]
        LHIP_last_hidden_state = self.base_model(LHIP_imgs_tensor)["last_hidden_state"]
        # Get logits
        img_logits = self.classifier(img_last_hidden_state[:, 0, :])
        RSJ_logits = self.classifier(RSJ_last_hidden_state[:, 0, :])
        LSJ_logits = self.classifier(LSJ_last_hidden_state[:, 0, :])
        RHIP_logits = self.classifier(RHIP_last_hidden_state[:, 0, :])
        LHIP_logits = self.classifier(LHIP_last_hidden_state[:, 0, :])

        all_last_hidden_states = torch.stack(
            [
                img_last_hidden_state[:, 0, :],
                RSJ_last_hidden_state[:, 0, :],
                LSJ_last_hidden_state[:, 0, :],
                RHIP_last_hidden_state[:, 0, :],
                LHIP_last_hidden_state[:, 0, :],
            ],
            dim=0,
        )

        
        all_logits = torch.stack(
            [
                img_logits,
                RSJ_logits,
                LSJ_logits,
                RHIP_logits,
                LHIP_logits,
            ],
            dim=0,
        )

        
        if self.method == "average":
            final_logits = torch.mean(all_logits, dim=0)
            final_last_hidden_states = torch.mean(all_last_hidden_states, dim=0)

        
        elif self.method == "attention":
            attention_scores = F.softmax(self.attention_weights, dim=0)  
            attention_scores = attention_scores.view(5, 1, 1)  
            # print("all_logits shape:", all_logits.shape)
            # print("attention_scores shape:", attention_scores.shape)
            final_logits = torch.sum(all_logits * attention_scores, dim=0)  
            final_last_hidden_states = torch.sum(
                all_last_hidden_states * attention_scores, dim=0
            )  

        # Custom logic can be added here
        return final_logits, final_last_hidden_states
    
    def grad_cam_predict(self, imgs_batch: ImgsData, labels=None, **kwargs):
        
        
        all_imgs_tensor = torch.cat(
            [
                imgs_batch.img,
                imgs_batch.RSJ_imgs,
                imgs_batch.LSJ_imgs,
                imgs_batch.RHIP_imgs,
                imgs_batch.LHIP_imgs,
            ],
            dim=0,  
        )

        
        all_last_hidden_states = self.base_model(all_imgs_tensor)["last_hidden_state"]

        
        all_logits = self.classifier(all_last_hidden_states[:, 0, :])

        
        img_size = imgs_batch.img.size(0)
        RSJ_size = imgs_batch.RSJ_imgs.size(0)
        LSJ_size = imgs_batch.LSJ_imgs.size(0)
        RHIP_size = imgs_batch.RHIP_imgs.size(0)
        LHIP_size = imgs_batch.LHIP_imgs.size(0)

        
        img_logits, RSJ_logits, LSJ_logits, RHIP_logits, LHIP_logits = torch.split(
            all_logits, [img_size, RSJ_size, LSJ_size, RHIP_size, LHIP_size], dim=0
        )
        img_last_hidden_state, RSJ_last_hidden_state, LSJ_last_hidden_state, RHIP_last_hidden_state, LHIP_last_hidden_state = torch.split(
            all_last_hidden_states, [img_size, RSJ_size, LSJ_size, RHIP_size, LHIP_size], dim=0
        )

        
        if self.method == "average":
            final_logits = torch.mean(all_logits, dim=0)
            final_last_hidden_states = torch.mean(all_last_hidden_states, dim=0)

        elif self.method == "attention":
            
            attention_scores = F.softmax(self.attention_weights, dim=0)  # [5]
            
            
            attention_scores_logits = attention_scores.view(5, 1).repeat(1, 5)  # [5, 5]
            final_logits = torch.sum(all_logits * attention_scores_logits, dim=0)  

            
            attention_scores_hidden = attention_scores.view(5, 1, 1)  # [5, 1, 1]
            final_last_hidden_states = torch.sum(all_last_hidden_states * attention_scores_hidden, dim=0)

        return final_logits, final_last_hidden_states



class CustomVitForMultiLabelImageClassificationSingleimg(AutoModelForImageClassification):
    def forward(self, imgs_batch: ImgsData, labels=None, **kwargs):
        # Your custom forward logic here
        imgs_tensor = imgs_batch.img
        # Get last hidden state
        img_last_hidden_state = self.base_model(imgs_tensor)["last_hidden_state"]
        # Get logits
        img_logits = self.classifier(img_last_hidden_state[:, 0, :])

        
        final_logits = img_logits

        # Comput loss
        loss = None
        # loss function
        loss_fn = nn.BCEWithLogitsLoss()
        # Comput loss
        loss = loss_fn(final_logits, labels)
        # Custom logic can be added here
        return ImageClassifierOutput(
            loss=loss,
            logits=final_logits,
            hidden_states=None,
            attentions=None,
        )

    def predict(self, imgs_batch: ImgsData, labels=None, **kwargs):
        # Your custom forward logic here
        imgs_tensor = imgs_batch.img

        # Get last hidden state
        img_last_hidden_state = self.base_model(imgs_tensor)["last_hidden_state"]

        # Get logits
        img_logits = self.classifier(img_last_hidden_state[:, 0, :])

        final_logits = img_logits
        final_last_hidden_states = img_last_hidden_state[:, 0, :]

        return final_logits, final_last_hidden_states


# "sail/poolformer_m48"
class CustomPoolformerForMultiLabelImageClassification(AutoModelForImageClassification):
    def forward(self, imgs_batch: ImgsData, labels=None, **kwargs):
        # Your custom forward logic here
        imgs_tensor = imgs_batch.img
        RSJ_imgs_tensor = imgs_batch.RSJ_imgs
        LSJ_imgs_tensor = imgs_batch.LSJ_imgs
        RHIP_imgs_tensor = imgs_batch.RHIP_imgs
        LHIP_imgs_tensor = imgs_batch.LHIP_imgs
        # Get last hidden state
        img_sequence_output = self.poolformer(imgs_tensor)[0]
        RSJ_sequence_output = self.poolformer(RSJ_imgs_tensor)[0]
        LSJ_sequence_output = self.poolformer(LSJ_imgs_tensor)[0]
        RHIP_sequence_output = self.poolformer(RHIP_imgs_tensor)[0]
        LHIP_sequence_output = self.poolformer(LHIP_imgs_tensor)[0]
        # Get logits
        img_logits = self.classifier(self.norm(img_sequence_output).mean([-2, -1]))
        RSJ_logits = self.classifier(self.norm(RSJ_sequence_output).mean([-2, -1]))
        LSJ_logits = self.classifier(self.norm(LSJ_sequence_output).mean([-2, -1]))
        RHIP_logits = self.classifier(self.norm(RHIP_sequence_output).mean([-2, -1]))
        LHIP_logits = self.classifier(self.norm(LHIP_sequence_output).mean([-2, -1]))

        
        all_logits = torch.stack(
            [
                img_logits,
                RSJ_logits,
                LSJ_logits,
                RHIP_logits,
                LHIP_logits,
            ],
            dim=0,
        )

        
        if self.method == "average":
            final_logits = torch.mean(all_logits, dim=0)

        
        elif self.method == "attention":
            attention_scores = F.softmax(self.attention_weights, dim=0)  
            attention_scores = attention_scores.view(5, 1, 1)  
            # print("all_logits shape:", all_logits.shape)
            # print("attention_scores shape:", attention_scores.shape)
            final_logits = torch.sum(all_logits * attention_scores, dim=0)  

        # Comput loss
        loss = None
        # loss function
        loss_fn = nn.BCEWithLogitsLoss()
        # Comput loss
        loss = loss_fn(final_logits, labels)
        # Custom logic can be added here
        return ImageClassifierOutput(
            loss=loss,
            logits=final_logits,
            hidden_states=None,
            attentions=None,
        )

    def predict(self, imgs_batch: ImgsData, labels=None, **kwargs):
        # Your custom forward logic here
        imgs_tensor = imgs_batch.img
        RSJ_imgs_tensor = imgs_batch.RSJ_imgs
        LSJ_imgs_tensor = imgs_batch.LSJ_imgs
        RHIP_imgs_tensor = imgs_batch.RHIP_imgs
        LHIP_imgs_tensor = imgs_batch.LHIP_imgs
        # Get last hidden state
        img_sequence_output = self.poolformer(imgs_tensor)[0]
        RSJ_sequence_output = self.poolformer(RSJ_imgs_tensor)[0]
        LSJ_sequence_output = self.poolformer(LSJ_imgs_tensor)[0]
        RHIP_sequence_output = self.poolformer(RHIP_imgs_tensor)[0]
        LHIP_sequence_output = self.poolformer(LHIP_imgs_tensor)[0]
        # Get logits
        img_logits = self.classifier(self.norm(img_sequence_output).mean([-2, -1]))
        RSJ_logits = self.classifier(self.norm(RSJ_sequence_output).mean([-2, -1]))
        LSJ_logits = self.classifier(self.norm(LSJ_sequence_output).mean([-2, -1]))
        RHIP_logits = self.classifier(self.norm(RHIP_sequence_output).mean([-2, -1]))
        LHIP_logits = self.classifier(self.norm(LHIP_sequence_output).mean([-2, -1]))

        all_sequence_outputs = torch.stack(
            [
                self.norm(img_sequence_output).mean([-2, -1]),
                self.norm(img_sequence_output).mean([-2, -1]),
                self.norm(img_sequence_output).mean([-2, -1]),
                self.norm(img_sequence_output).mean([-2, -1]),
                self.norm(img_sequence_output).mean([-2, -1]),
            ],
            dim=0,
        )

        
        all_logits = torch.stack(
            [
                img_logits,
                RSJ_logits,
                LSJ_logits,
                RHIP_logits,
                LHIP_logits,
            ],
            dim=0,
        )

        
        if self.method == "average":
            final_logits = torch.mean(all_logits, dim=0)
            final_sequence_outputs = torch.mean(all_sequence_outputs, dim=0)

        
        elif self.method == "attention":
            attention_scores = F.softmax(self.attention_weights, dim=0)  
            attention_scores = attention_scores.view(5, 1, 1)  
            # print("all_logits shape:", all_logits.shape)
            # print("attention_scores shape:", attention_scores.shape)
            attention_scores_expanded = attention_scores
            final_logits = torch.sum(all_logits * attention_scores, dim=0)  
            final_sequence_outputs = torch.sum(
                all_sequence_outputs * attention_scores_expanded, dim=0
            )  

        # Custom logic can be added here
        return final_logits, final_sequence_outputs
    
    def grad_cam_predict(self, imgs_batch: ImgsData, labels=None, **kwargs):
        
        
        all_imgs_tensor = torch.cat(
            [
                imgs_batch.img,
                imgs_batch.RSJ_imgs,
                imgs_batch.LSJ_imgs,
                imgs_batch.RHIP_imgs,
                imgs_batch.LHIP_imgs,
            ],
            dim=0,  
        )

        
        all_sequence_outputs = self.poolformer(all_imgs_tensor)[0]  # [batch_size, height, width, channels]

        
        normalized_sequence_outputs = self.norm(all_sequence_outputs).mean([-2, -1])  # [batch_size, channels]

        
        all_logits = self.classifier(normalized_sequence_outputs)  # [batch_size, num_classes]

        
        img_size = imgs_batch.img.size(0)
        RSJ_size = imgs_batch.RSJ_imgs.size(0)
        LSJ_size = imgs_batch.LSJ_imgs.size(0)
        RHIP_size = imgs_batch.RHIP_imgs.size(0)
        LHIP_size = imgs_batch.LHIP_imgs.size(0)

        
        img_logits, RSJ_logits, LSJ_logits, RHIP_logits, LHIP_logits = torch.split(
            all_logits, [img_size, RSJ_size, LSJ_size, RHIP_size, LHIP_size], dim=0
        )
        img_normalized_sequence, RSJ_normalized_sequence, LSJ_normalized_sequence, RHIP_normalized_sequence, LHIP_normalized_sequence = torch.split(
            normalized_sequence_outputs, [img_size, RSJ_size, LSJ_size, RHIP_size, LHIP_size], dim=0
        )

        
        if self.method == "average":
            
            final_logits = torch.mean(all_logits, dim=0)  
            final_sequence_outputs = torch.mean(normalized_sequence_outputs, dim=0)

        elif self.method == "attention":
            
            attention_scores = F.softmax(self.attention_weights, dim=0)  # [5]
            
            
            attention_scores_logits = attention_scores.view(5, 1).repeat(1, all_logits.shape[1])  # [5, num_classes]
            final_logits = torch.sum(all_logits * attention_scores_logits, dim=0)  
            
            attention_scores_sequence = attention_scores.view(5, 1).repeat(1, normalized_sequence_outputs.shape[1])  # [5, channels]
            final_sequence_outputs = torch.sum(normalized_sequence_outputs * attention_scores_sequence, dim=0)

        else:
            
            final_logits = all_logits
            final_sequence_outputs = normalized_sequence_outputs

        return final_logits, final_sequence_outputs



class CustomPoolformerForMultiLabelImageClassificationSingleimg(AutoModelForImageClassification):
    def forward(self, imgs_batch: ImgsData, labels=None, **kwargs):
        # Your custom forward logic here
        imgs_tensor = imgs_batch.img
        # Get last hidden state
        img_sequence_output = self.poolformer(imgs_tensor)[0]
        # Get logits
        img_logits = self.classifier(self.norm(img_sequence_output).mean([-2, -1]))

        # Comput loss
        loss = None
        # loss function
        loss_fn = nn.BCEWithLogitsLoss()
        # Comput loss
        loss = loss_fn(img_logits, labels)
        # Custom logic can be added here
        return ImageClassifierOutput(
            loss=loss,
            logits=img_logits,
            hidden_states=None,
            attentions=None,
        )

    def predict(self, imgs_batch: ImgsData, labels=None, **kwargs):
        # Your custom forward logic here
        imgs_tensor = imgs_batch.img

        # Get last hidden state
        img_sequence_output = self.poolformer(imgs_tensor)[0]

        # Get logits
        img_logits = self.classifier(self.norm(img_sequence_output).mean([-2, -1]))

        final_logits = img_logits
        final_sequence_outputs = self.norm(img_sequence_output).mean([-2, -1])

        return final_logits, final_sequence_outputs


# convnext "facebook/convnext-base-224"
class CustomConvnextForMultiLabelImageClassification(AutoModelForImageClassification):
    def forward(
        self,
        imgs_batch: ImgsData,
        labels=None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Your custom forward logic here
        imgs_tensor = imgs_batch.img
        RSJ_imgs_tensor = imgs_batch.RSJ_imgs
        LSJ_imgs_tensor = imgs_batch.LSJ_imgs
        RHIP_imgs_tensor = imgs_batch.RHIP_imgs
        LHIP_imgs_tensor = imgs_batch.LHIP_imgs
        # Get outputs
        img_outputs = self.convnext(
            imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        RSJ_outputs = self.convnext(
            RSJ_imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        LSJ_outputs = self.convnext(
            LSJ_imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        RHIP_outputs = self.convnext(
            RHIP_imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        LHIP_outputs = self.convnext(
            LHIP_imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        # Get pooled output
        img_pooled_output = img_outputs.pooler_output if return_dict else img_outputs[1]
        RSJ_pooled_output = RSJ_outputs.pooler_output if return_dict else RSJ_outputs[1]
        LSJ_pooled_output = LSJ_outputs.pooler_output if return_dict else LSJ_outputs[1]
        RHIP_pooled_output = RHIP_outputs.pooler_output if return_dict else RHIP_outputs[1]
        LHIP_pooled_output = LHIP_outputs.pooler_output if return_dict else LHIP_outputs[1]

        # Get logits
        img_logits = self.classifier(img_pooled_output)
        RSJ_logits = self.classifier(RSJ_pooled_output)
        LSJ_logits = self.classifier(LSJ_pooled_output)
        RHIP_logits = self.classifier(RHIP_pooled_output)
        LHIP_logits = self.classifier(LHIP_pooled_output)

        
        all_logits = torch.stack(
            [
                img_logits,
                RSJ_logits,
                LSJ_logits,
                RHIP_logits,
                LHIP_logits,
            ],
            dim=0,
        )

        
        if self.method == "average":
            final_logits = torch.mean(all_logits, dim=0)

        
        elif self.method == "attention":
            attention_scores = F.softmax(self.attention_weights, dim=0)  
            attention_scores = attention_scores.view(5, 1, 1)  
            # print("all_logits shape:", all_logits.shape)
            # print("attention_scores shape:", attention_scores.shape)
            final_logits = torch.sum(all_logits * attention_scores, dim=0)  

        # Comput loss
        loss = None
        # loss function
        loss_fn = nn.BCEWithLogitsLoss()
        # Comput loss
        loss = loss_fn(final_logits, labels)
        # Custom logic can be added here
        return ImageClassifierOutput(
            loss=loss,
            logits=final_logits,
            hidden_states=None,
        )

    def predict(
        self,
        imgs_batch: ImgsData,
        labels=None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Your custom forward logic here
        imgs_tensor = imgs_batch.img
        RSJ_imgs_tensor = imgs_batch.RSJ_imgs
        LSJ_imgs_tensor = imgs_batch.LSJ_imgs
        RHIP_imgs_tensor = imgs_batch.RHIP_imgs
        LHIP_imgs_tensor = imgs_batch.LHIP_imgs
        # Get outputs
        img_outputs = self.convnext(
            imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        RSJ_outputs = self.convnext(
            RSJ_imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        LSJ_outputs = self.convnext(
            LSJ_imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        RHIP_outputs = self.convnext(
            RHIP_imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        LHIP_outputs = self.convnext(
            LHIP_imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        # Get pooled output
        img_pooled_output = img_outputs.pooler_output if return_dict else img_outputs[1]
        RSJ_pooled_output = RSJ_outputs.pooler_output if return_dict else RSJ_outputs[1]
        LSJ_pooled_output = LSJ_outputs.pooler_output if return_dict else LSJ_outputs[1]
        RHIP_pooled_output = RHIP_outputs.pooler_output if return_dict else RHIP_outputs[1]
        LHIP_pooled_output = LHIP_outputs.pooler_output if return_dict else LHIP_outputs[1]

        all_pooled_output = torch.stack(
            [
                img_pooled_output,
                img_pooled_output,
                img_pooled_output,
                img_pooled_output,
                img_pooled_output,
            ],
            dim=0,
        )

        # Get logits
        img_logits = self.classifier(img_pooled_output)
        RSJ_logits = self.classifier(RSJ_pooled_output)
        LSJ_logits = self.classifier(LSJ_pooled_output)
        RHIP_logits = self.classifier(RHIP_pooled_output)
        LHIP_logits = self.classifier(LHIP_pooled_output)

        
        all_logits = torch.stack(
            [
                img_logits,
                RSJ_logits,
                LSJ_logits,
                RHIP_logits,
                LHIP_logits,
            ],
            dim=0,
        )

        
        if self.method == "average":
            final_logits = torch.mean(all_logits, dim=0)
            final_pooled_output = torch.mean(all_pooled_output, dim=0)

        
        elif self.method == "attention":
            attention_scores = F.softmax(self.attention_weights, dim=0)  
            attention_scores = attention_scores.view(5, 1, 1)  
            # print("all_logits shape:", all_logits.shape)
            # print("attention_scores shape:", attention_scores.shape)
            final_logits = torch.sum(all_logits * attention_scores, dim=0)  
            final_pooled_output = torch.sum(all_pooled_output * attention_scores, dim=0)  

        # Custom logic can be added here
        return final_logits, final_pooled_output


class CustomConvnextForMultiLabelImageClassificationSingleimg(AutoModelForImageClassification):
    def forward(
        self,
        imgs_batch: ImgsData,
        labels=None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Your custom forward logic here
        imgs_tensor = imgs_batch.img
        # Get outputs
        img_outputs = self.convnext(
            imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        # Get pooled output
        img_pooled_output = img_outputs.pooler_output if return_dict else img_outputs[1]

        # Get logits
        img_logits = self.classifier(img_pooled_output)

        # Comput loss
        loss = None
        # loss function
        loss_fn = nn.BCEWithLogitsLoss()
        # Comput loss
        loss = loss_fn(img_logits, labels)
        # Custom logic can be added here
        return ImageClassifierOutput(
            loss=loss,
            logits=img_logits,
            hidden_states=None,
        )

    def predict(
        self,
        imgs_batch: ImgsData,
        labels=None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Your custom forward logic here
        imgs_tensor = imgs_batch.img

        # Get outputs
        img_outputs = self.convnext(
            imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )

        # Get pooled output
        img_pooled_output = img_outputs.pooler_output if return_dict else img_outputs[1]

        # Get logits
        img_logits = self.classifier(img_pooled_output)

        final_logits = img_logits
        final_pooled_output = img_pooled_output

        # Custom logic can be added here
        return final_logits, final_pooled_output


# convnextv2 "facebook/convnextv2-base-1k-224"
class CustomConvnextv2ForMultiLabelImageClassification(AutoModelForImageClassification):
    def forward(
        self,
        imgs_batch: ImgsData,
        labels=None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Your custom forward logic here
        imgs_tensor = imgs_batch.img
        RSJ_imgs_tensor = imgs_batch.RSJ_imgs
        LSJ_imgs_tensor = imgs_batch.LSJ_imgs
        RHIP_imgs_tensor = imgs_batch.RHIP_imgs
        LHIP_imgs_tensor = imgs_batch.LHIP_imgs
        # Get outputs
        img_outputs = self.convnextv2(
            imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        RSJ_outputs = self.convnextv2(
            RSJ_imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        LSJ_outputs = self.convnextv2(
            LSJ_imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        RHIP_outputs = self.convnextv2(
            RHIP_imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        LHIP_outputs = self.convnextv2(
            LHIP_imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        # Get pooled output
        img_pooled_output = img_outputs.pooler_output if return_dict else img_outputs[1]
        RSJ_pooled_output = RSJ_outputs.pooler_output if return_dict else RSJ_outputs[1]
        LSJ_pooled_output = LSJ_outputs.pooler_output if return_dict else LSJ_outputs[1]
        RHIP_pooled_output = RHIP_outputs.pooler_output if return_dict else RHIP_outputs[1]
        LHIP_pooled_output = LHIP_outputs.pooler_output if return_dict else LHIP_outputs[1]

        # Get logits
        img_logits = self.classifier(img_pooled_output)
        RSJ_logits = self.classifier(RSJ_pooled_output)
        LSJ_logits = self.classifier(LSJ_pooled_output)
        RHIP_logits = self.classifier(RHIP_pooled_output)
        LHIP_logits = self.classifier(LHIP_pooled_output)

        
        all_logits = torch.stack(
            [
                img_logits,
                RSJ_logits,
                LSJ_logits,
                RHIP_logits,
                LHIP_logits,
            ],
            dim=0,
        )

        
        if self.method == "average":
            final_logits = torch.mean(all_logits, dim=0)

        
        elif self.method == "attention":
            attention_scores = F.softmax(self.attention_weights, dim=0)  
            attention_scores = attention_scores.view(5, 1, 1)  
            # print("all_logits shape:", all_logits.shape)
            # print("attention_scores shape:", attention_scores.shape)
            final_logits = torch.sum(all_logits * attention_scores, dim=0)  

        # Comput loss
        loss = None
        # loss function
        loss_fn = nn.BCEWithLogitsLoss()
        # Comput loss
        loss = loss_fn(final_logits, labels)
        # Custom logic can be added here
        return ImageClassifierOutput(
            loss=loss,
            logits=final_logits,
            hidden_states=None,
        )

    def predict(
        self,
        imgs_batch: ImgsData,
        labels=None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Your custom forward logic here
        imgs_tensor = imgs_batch.img
        RSJ_imgs_tensor = imgs_batch.RSJ_imgs
        LSJ_imgs_tensor = imgs_batch.LSJ_imgs
        RHIP_imgs_tensor = imgs_batch.RHIP_imgs
        LHIP_imgs_tensor = imgs_batch.LHIP_imgs
        # Get outputs
        img_outputs = self.convnextv2(
            imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        RSJ_outputs = self.convnextv2(
            RSJ_imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        LSJ_outputs = self.convnextv2(
            LSJ_imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        RHIP_outputs = self.convnextv2(
            RHIP_imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        LHIP_outputs = self.convnextv2(
            LHIP_imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        # Get pooled output
        img_pooled_output = img_outputs.pooler_output if return_dict else img_outputs[1]
        RSJ_pooled_output = RSJ_outputs.pooler_output if return_dict else RSJ_outputs[1]
        LSJ_pooled_output = LSJ_outputs.pooler_output if return_dict else LSJ_outputs[1]
        RHIP_pooled_output = RHIP_outputs.pooler_output if return_dict else RHIP_outputs[1]
        LHIP_pooled_output = LHIP_outputs.pooler_output if return_dict else LHIP_outputs[1]

        all_pooled_output = torch.stack(
            [
                img_pooled_output,
                img_pooled_output,
                img_pooled_output,
                img_pooled_output,
                img_pooled_output,
            ],
            dim=0,
        )

        # Get logits
        img_logits = self.classifier(img_pooled_output)
        RSJ_logits = self.classifier(RSJ_pooled_output)
        LSJ_logits = self.classifier(LSJ_pooled_output)
        RHIP_logits = self.classifier(RHIP_pooled_output)
        LHIP_logits = self.classifier(LHIP_pooled_output)

        
        all_logits = torch.stack(
            [
                img_logits,
                RSJ_logits,
                LSJ_logits,
                RHIP_logits,
                LHIP_logits,
            ],
            dim=0,
        )

        
        if self.method == "average":
            final_logits = torch.mean(all_logits, dim=0)
            final_pooled_output = torch.mean(all_pooled_output, dim=0)

        
        elif self.method == "attention":
            attention_scores = F.softmax(self.attention_weights, dim=0)  
            attention_scores = attention_scores.view(5, 1, 1)  
            # print("all_logits shape:", all_logits.shape)
            # print("attention_scores shape:", attention_scores.shape)
            final_logits = torch.sum(all_logits * attention_scores, dim=0)  
            final_pooled_output = torch.sum(all_pooled_output * attention_scores, dim=0)  

        # Custom logic can be added here
        return final_logits, final_pooled_output


class CustomConvnextv2ForMultiLabelImageClassificationSingleimg(AutoModelForImageClassification):
    def forward(
        self,
        imgs_batch: ImgsData,
        labels=None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Your custom forward logic here
        imgs_tensor = imgs_batch.img
        # Get outputs
        img_outputs = self.convnextv2(
            imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        # Get pooled output
        img_pooled_output = img_outputs.pooler_output if return_dict else img_outputs[1]

        # Get logits
        img_logits = self.classifier(img_pooled_output)

        # Comput loss
        loss = None
        # loss function
        loss_fn = nn.BCEWithLogitsLoss()
        # Comput loss
        loss = loss_fn(img_logits, labels)
        # Custom logic can be added here
        return ImageClassifierOutput(
            loss=loss,
            logits=img_logits,
            hidden_states=None,
        )

    def predict(
        self,
        imgs_batch: ImgsData,
        labels=None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Your custom forward logic here
        imgs_tensor = imgs_batch.img

        # Get outputs
        img_outputs = self.convnextv2(
            imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )

        # Get pooled output
        img_pooled_output = img_outputs.pooler_output if return_dict else img_outputs[1]

        # Get logits
        img_logits = self.classifier(img_pooled_output)

        final_logits = img_logits
        final_pooled_output = img_pooled_output

        # Custom logic can be added here
        return final_logits, final_pooled_output


# resnet "microsoft/resnet-50"
class CustomResNetForMultiLabelImageClassification(AutoModelForImageClassification):
    def forward(
        self,
        imgs_batch: ImgsData,
        labels=None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Your custom forward logic here
        imgs_tensor = imgs_batch.img
        RSJ_imgs_tensor = imgs_batch.RSJ_imgs
        LSJ_imgs_tensor = imgs_batch.LSJ_imgs
        RHIP_imgs_tensor = imgs_batch.RHIP_imgs
        LHIP_imgs_tensor = imgs_batch.LHIP_imgs
        # Get outputs
        img_outputs = self.resnet(
            imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        RSJ_outputs = self.resnet(
            RSJ_imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        LSJ_outputs = self.resnet(
            LSJ_imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        RHIP_outputs = self.resnet(
            RHIP_imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        LHIP_outputs = self.resnet(
            LHIP_imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        # Get pooled output
        img_pooled_output = img_outputs.pooler_output if return_dict else img_outputs[1]
        RSJ_pooled_output = RSJ_outputs.pooler_output if return_dict else RSJ_outputs[1]
        LSJ_pooled_output = LSJ_outputs.pooler_output if return_dict else LSJ_outputs[1]
        RHIP_pooled_output = RHIP_outputs.pooler_output if return_dict else RHIP_outputs[1]
        LHIP_pooled_output = LHIP_outputs.pooler_output if return_dict else LHIP_outputs[1]

        # Get logits
        img_logits = self.classifier(img_pooled_output)
        RSJ_logits = self.classifier(RSJ_pooled_output)
        LSJ_logits = self.classifier(LSJ_pooled_output)
        RHIP_logits = self.classifier(RHIP_pooled_output)
        LHIP_logits = self.classifier(LHIP_pooled_output)

        
        all_logits = torch.stack(
            [
                img_logits,
                RSJ_logits,
                LSJ_logits,
                RHIP_logits,
                LHIP_logits,
            ],
            dim=0,
        )

        
        if self.method == "average":
            final_logits = torch.mean(all_logits, dim=0)

        
        elif self.method == "attention":
            attention_scores = F.softmax(self.attention_weights, dim=0)  
            attention_scores = attention_scores.view(5, 1, 1)  
            # print("all_logits shape:", all_logits.shape)
            # print("attention_scores shape:", attention_scores.shape)
            final_logits = torch.sum(all_logits * attention_scores, dim=0)  

        # Comput loss
        loss = None
        # loss function
        loss_fn = nn.BCEWithLogitsLoss()
        # Comput loss
        loss = loss_fn(final_logits, labels)
        # Custom logic can be added here
        return ImageClassifierOutput(
            loss=loss,
            logits=final_logits,
            hidden_states=None,
        )

    def predict(
        self,
        imgs_batch: ImgsData,
        labels=None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Your custom forward logic here
        imgs_tensor = imgs_batch.img
        RSJ_imgs_tensor = imgs_batch.RSJ_imgs
        LSJ_imgs_tensor = imgs_batch.LSJ_imgs
        RHIP_imgs_tensor = imgs_batch.RHIP_imgs
        LHIP_imgs_tensor = imgs_batch.LHIP_imgs
        # Get outputs
        img_outputs = self.resnet(
            imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        RSJ_outputs = self.resnet(
            RSJ_imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        LSJ_outputs = self.resnet(
            LSJ_imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        RHIP_outputs = self.resnet(
            RHIP_imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        LHIP_outputs = self.resnet(
            LHIP_imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        # Get pooled output
        img_pooled_output = img_outputs.pooler_output if return_dict else img_outputs[1]
        RSJ_pooled_output = RSJ_outputs.pooler_output if return_dict else RSJ_outputs[1]
        LSJ_pooled_output = LSJ_outputs.pooler_output if return_dict else LSJ_outputs[1]
        RHIP_pooled_output = RHIP_outputs.pooler_output if return_dict else RHIP_outputs[1]
        LHIP_pooled_output = LHIP_outputs.pooler_output if return_dict else LHIP_outputs[1]

        all_pooled_output = torch.stack(
            [
                img_pooled_output,
                img_pooled_output,
                img_pooled_output,
                img_pooled_output,
                img_pooled_output,
            ],
            dim=0,
        )

        # Get logits
        img_logits = self.classifier(img_pooled_output)
        RSJ_logits = self.classifier(RSJ_pooled_output)
        LSJ_logits = self.classifier(LSJ_pooled_output)
        RHIP_logits = self.classifier(RHIP_pooled_output)
        LHIP_logits = self.classifier(LHIP_pooled_output)

        
        all_logits = torch.stack(
            [
                img_logits,
                RSJ_logits,
                LSJ_logits,
                RHIP_logits,
                LHIP_logits,
            ],
            dim=0,
        )

        
        if self.method == "average":
            final_logits = torch.mean(all_logits, dim=0)
            final_pooled_output = torch.mean(all_pooled_output, dim=0)

        
        elif self.method == "attention":
            attention_scores = F.softmax(self.attention_weights, dim=0)  
            attention_scores = attention_scores.view(5, 1, 1)  
            # print("all_logits shape:", all_logits.shape)
            # print("attention_scores shape:", attention_scores.shape)
            attention_scores_expanded = attention_scores.unsqueeze(-1)
            final_logits = torch.sum(all_logits * attention_scores, dim=0)  
            final_pooled_output = torch.sum(all_pooled_output * attention_scores_expanded, dim=0)  

        # Custom logic can be added here
        return final_logits, final_pooled_output


class CustomResNetForMultiLabelImageClassificationSingleimg(AutoModelForImageClassification):
    def forward(
        self,
        imgs_batch: ImgsData,
        labels=None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Your custom forward logic here
        imgs_tensor = imgs_batch.img
        # Get outputs
        img_outputs = self.resnet(
            imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        # Get pooled output
        img_pooled_output = img_outputs.pooler_output if return_dict else img_outputs[1]

        # Get logits
        img_logits = self.classifier(img_pooled_output)

        # Comput loss
        loss = None
        # loss function
        loss_fn = nn.BCEWithLogitsLoss()
        # Comput loss
        loss = loss_fn(img_logits, labels)
        # Custom logic can be added here
        return ImageClassifierOutput(
            loss=loss,
            logits=img_logits,
            hidden_states=None,
        )

    def predict(
        self,
        imgs_batch: ImgsData,
        labels=None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Your custom forward logic here
        imgs_tensor = imgs_batch.img

        # Get outputs
        img_outputs = self.resnet(
            imgs_tensor, output_hidden_states=output_hidden_states, return_dict=return_dict
        )

        # Get pooled output
        img_pooled_output = img_outputs.pooler_output if return_dict else img_outputs[1]

        # Get logits
        img_logits = self.classifier(img_pooled_output)

        final_logits = img_logits
        final_pooled_output = img_pooled_output

        # Custom logic can be added here
        return final_logits, final_pooled_output


# Swin "microsoft/swin-base-patch4-window7-224"
class CustomSwinForMultiLabelImageClassification(AutoModelForImageClassification):
    def forward(
        self,
        imgs_batch: ImgsData,
        labels=None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Your custom forward logic here
        imgs_tensor = imgs_batch.img
        RSJ_imgs_tensor = imgs_batch.RSJ_imgs
        LSJ_imgs_tensor = imgs_batch.LSJ_imgs
        RHIP_imgs_tensor = imgs_batch.RHIP_imgs
        LHIP_imgs_tensor = imgs_batch.LHIP_imgs
        # Get outputs
        img_outputs = self.swin(
            imgs_tensor,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )
        RSJ_outputs = self.swin(
            RSJ_imgs_tensor,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )
        LSJ_outputs = self.swin(
            LSJ_imgs_tensor,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )
        RHIP_outputs = self.swin(
            RHIP_imgs_tensor,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )
        LHIP_outputs = self.swin(
            LHIP_imgs_tensor,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        # Get pooled output
        img_pooled_output = img_outputs.pooler_output if return_dict else img_outputs[1]
        RSJ_pooled_output = RSJ_outputs.pooler_output if return_dict else RSJ_outputs[1]
        LSJ_pooled_output = LSJ_outputs.pooler_output if return_dict else LSJ_outputs[1]
        RHIP_pooled_output = RHIP_outputs.pooler_output if return_dict else RHIP_outputs[1]
        LHIP_pooled_output = LHIP_outputs.pooler_output if return_dict else LHIP_outputs[1]

        # Get logits
        img_logits = self.classifier(img_pooled_output)
        RSJ_logits = self.classifier(RSJ_pooled_output)
        LSJ_logits = self.classifier(LSJ_pooled_output)
        RHIP_logits = self.classifier(RHIP_pooled_output)
        LHIP_logits = self.classifier(LHIP_pooled_output)

        
        all_logits = torch.stack(
            [
                img_logits,
                RSJ_logits,
                LSJ_logits,
                RHIP_logits,
                LHIP_logits,
            ],
            dim=0,
        )

        
        if self.method == "average":
            final_logits = torch.mean(all_logits, dim=0)

        elif self.method == "attention":
            attention_scores = F.softmax(self.attention_weights, dim=0)  # 使用Softmax将权重归一化
            attention_scores = attention_scores.view(5, 1, 1)  # 调整维度
            # print("all_logits shape:", all_logits.shape)
            # print("attention_scores shape:", attention_scores.shape)
            final_logits = torch.sum(all_logits * attention_scores, dim=0)  # 加权平均

        # Comput loss
        loss = None
        # loss function
        loss_fn = nn.BCEWithLogitsLoss()
        # Comput loss
        loss = loss_fn(final_logits, labels)
        # Custom logic can be added here
        return ImageClassifierOutput(
            loss=loss,
            logits=final_logits,
            hidden_states=None,
        )

    def predict(
        self,
        imgs_batch: ImgsData,
        labels=None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Your custom forward logic here
        imgs_tensor = imgs_batch.img
        RSJ_imgs_tensor = imgs_batch.RSJ_imgs
        LSJ_imgs_tensor = imgs_batch.LSJ_imgs
        RHIP_imgs_tensor = imgs_batch.RHIP_imgs
        LHIP_imgs_tensor = imgs_batch.LHIP_imgs
        # Get outputs
        # Get outputs
        img_outputs = self.swin(
            imgs_tensor,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )
        RSJ_outputs = self.swin(
            RSJ_imgs_tensor,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )
        LSJ_outputs = self.swin(
            LSJ_imgs_tensor,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )
        RHIP_outputs = self.swin(
            RHIP_imgs_tensor,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )
        LHIP_outputs = self.swin(
            LHIP_imgs_tensor,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )
        # Get pooled output
        img_pooled_output = img_outputs.pooler_output if return_dict else img_outputs[1]
        RSJ_pooled_output = RSJ_outputs.pooler_output if return_dict else RSJ_outputs[1]
        LSJ_pooled_output = LSJ_outputs.pooler_output if return_dict else LSJ_outputs[1]
        RHIP_pooled_output = RHIP_outputs.pooler_output if return_dict else RHIP_outputs[1]
        LHIP_pooled_output = LHIP_outputs.pooler_output if return_dict else LHIP_outputs[1]

        all_pooled_output = torch.stack(
            [
                img_pooled_output,
                img_pooled_output,
                img_pooled_output,
                img_pooled_output,
                img_pooled_output,
            ],
            dim=0,
        )

        # Get logits
        img_logits = self.classifier(img_pooled_output)
        RSJ_logits = self.classifier(RSJ_pooled_output)
        LSJ_logits = self.classifier(LSJ_pooled_output)
        RHIP_logits = self.classifier(RHIP_pooled_output)
        LHIP_logits = self.classifier(LHIP_pooled_output)

        all_logits = torch.stack(
            [
                img_logits,
                RSJ_logits,
                LSJ_logits,
                RHIP_logits,
                LHIP_logits,
            ],
            dim=0,
        )

        if self.method == "average":
            final_logits = torch.mean(all_logits, dim=0)
            final_pooled_output = torch.mean(all_pooled_output, dim=0)

        elif self.method == "attention":
            attention_scores = F.softmax(self.attention_weights, dim=0)  
            attention_scores = attention_scores.view(5, 1, 1) 
            # print("all_logits shape:", all_logits.shape)
            # print("attention_scores shape:", attention_scores.shape)
            final_logits = torch.sum(all_logits * attention_scores, dim=0) 
            final_pooled_output = torch.sum(all_pooled_output * attention_scores, dim=0)  

        # Custom logic can be added here
        return final_logits, final_pooled_output


class CustomSwinForMultiLabelImageClassificationSingleimg(AutoModelForImageClassification):
    def forward(
        self,
        imgs_batch: ImgsData,
        labels=None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Your custom forward logic here
        imgs_tensor = imgs_batch.img
        # Get outputs
        img_outputs = self.swin(
            imgs_tensor,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        # Get pooled output
        img_pooled_output = img_outputs.pooler_output if return_dict else img_outputs[1]

        # Get logits
        img_logits = self.classifier(img_pooled_output)

        # Comput loss
        loss = None
        # loss function
        loss_fn = nn.BCEWithLogitsLoss()
        # Comput loss
        loss = loss_fn(img_logits, labels)
        # Custom logic can be added here
        return ImageClassifierOutput(
            loss=loss,
            logits=img_logits,
            hidden_states=None,
        )

    def predict(
        self,
        imgs_batch: ImgsData,
        labels=None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Your custom forward logic here
        imgs_tensor = imgs_batch.img

        # Get outputs
        img_outputs = self.swin(
            imgs_tensor,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        # Get pooled output
        img_pooled_output = img_outputs.pooler_output if return_dict else img_outputs[1]

        # Get logits
        img_logits = self.classifier(img_pooled_output)

        final_logits, final_pooled_output = img_logits, img_pooled_output
        # Custom logic can be added here
        return final_logits, final_pooled_output
