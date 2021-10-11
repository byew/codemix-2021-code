
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, MultiLabelSoftMarginLoss
from .loss import FocalLoss, LabelSmoothingLoss
import torch.nn.functional as F

import sys
from transformers import BertPreTrainedModel, BertModel




class bert_cls_model(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        config.output_hidden_states = True
        self.bert = BertModel(config)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.weights = nn.Parameter(torch.rand(13, 1))
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        self.fc = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        batch_size = input_ids.shape[0]
        ht_cls = torch.cat(outputs[2])[:, :1, :].view(
            13, batch_size, 1, 768)

        atten = torch.sum(ht_cls * self.weights.view(
            13, 1, 1, 1), dim=[1, 3])
        atten = F.softmax(atten.view(-1), dim=0)
        feature = torch.sum(ht_cls * atten.view(13, 1, 1, 1), dim=[0, 2])

        if labels is not None:

            for i, dropout in enumerate(self.dropouts):
                if i == 0:
                    h = self.fc(dropout(feature))
                    outputs = (h,) + outputs[2:]  # add hidden states and attention if they are here

                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(h.view(-1, self.num_labels), labels.view(-1))
                else:
                    h += self.fc(dropout(feature))
                    outputs = (h,) + outputs[2:]  # add hidden states and attention if they are here
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(h.view(-1, self.num_labels), labels.view(-1))

            loss = loss / len(self.dropouts)
            outputs = (loss,) + outputs
        else:
            for i, dropout in enumerate(self.dropouts):
                if i == 0:
                    h = self.fc(dropout(feature))
                    outputs = (h,) + outputs[2:]  # add hidden states and attention if they are here
                else:
                    h += self.fc(dropout(feature))
                    outputs = (h,) + outputs[2:]  # add hidden states and attention if they are here

            outputs = outputs

        return outputs



class bertmodel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                loss = loss * 100
            outputs = [loss, ]
            outputs = outputs + [nn.functional.softmax(logits, -1)]
        else:
            outputs = nn.functional.softmax(logits, -1)

        return outputs  # (loss), logits, (hidden_states), (attentions)







class covid_rank6_model(BertPreTrainedModel):
    def __init__(self, config, pretrain_model_path=None, add_edit_dist=False, smoothing=0.05):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.linear1 = nn.Linear(config.hidden_size, self.config.num_labels)
        print(config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.reset_parameters() # 层初始化
        self.label_smooth_loss = LabelSmoothingLoss(classes=self.num_labels, smoothing=smoothing)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0.0)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        bert_mask = torch.ne(input_ids, 0)  # 找到 inputs 中不等于 0 的地方，置为 1（0表示padding）

        bert_enc = outputs[0]

        bert_enc = self.dropout(bert_enc)

        ##### 3.10 大 bug 修复：取 mean 之前，应该先把 padding 部分的特征去除！！！
        mask_2 = bert_mask  # 其余等于 1 的部分，即有效的部分
        mask_2_expand = mask_2.unsqueeze_(-1).expand(bert_enc.size()).float()
        sum_mask = mask_2_expand.sum(dim=1)  # 有效的部分“长度”求和
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        bert_enc = torch.sum(bert_enc * mask_2_expand, dim=1) / sum_mask

        logits = self.linear1(bert_enc)

        # 3.5 add label smoothing


        # outputs = (logits,) + outputs[2:]
        #
        # if labels is not None:
        #     if self.num_labels == 1:
        #         #  We are doing regression
        #         loss_fct = MSELoss()
        #         loss = loss_fct(logits.view(-1), labels.view(-1))
        #     else:
        #         loss = self.FocalLoss(logits, labels.view(-1))
        #         # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        #     outputs = (loss,) + outputs
        #
        # return outputs  # (loss), logits, (hidden_states), (attentions)
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = [loss, ]
            outputs = outputs + [nn.functional.softmax(logits, -1)]
        else:
            outputs = nn.functional.softmax(logits, -1)

        return outputs  # (loss), logits, (hidden_states), (attentions)


