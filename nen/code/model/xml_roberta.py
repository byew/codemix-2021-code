
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, MultiLabelSoftMarginLoss
from .loss import FocalLoss, LabelSmoothingLoss
import torch.nn.functional as F

import sys
from transformers import BertPreTrainedModel, BertModel
from transformers import XLMRobertaConfig, RobertaConfig, RobertaModel


def compute_kl_loss( p, q, pad_mask = None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

# pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

# You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss


# class RobertaForSequenceClassification(BertPreTrainedModel):
#     config_class = RobertaConfig
#     base_model_prefix = "roberta"
#
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         config.output_hidden_states=True
#
#         self.roberta = RobertaModel(config)
#         self.fc = nn.Linear(768, self.config.num_labels)
#
#         for param in self.roberta.parameters():
#             param.requires_grad = True
#
#         self.weights = nn.Parameter(torch.rand(13, 1))
#         self.dropouts = nn.ModuleList([
#             nn.Dropout(0.5) for _ in range(5)
#         ])
#         self.init_weights()
#
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#     ):
#
#         outputs= self.roberta(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )
#
#
#         batch_size = input_ids.shape[0]
#         ht_cls = torch.cat(outputs[2])[:, :1, :].view(
#             13, batch_size, 1, 768)
#
#         atten = torch.sum(ht_cls * self.weights.view(
#             13, 1, 1, 1), dim=[1, 3])
#         atten = F.softmax(atten.view(-1), dim=0)
#         feature = torch.sum(ht_cls * atten.view(13, 1, 1, 1), dim=[0, 2])
#
#         feature2 = torch.sum(ht_cls * atten.view(13, 1, 1, 1), dim=[0, 2])
#
#
#         if labels is not None:
#
#             for i, dropout in enumerate(self.dropouts):
#                 if i == 0:
#                     h = self.fc(dropout(feature))
#                     outputs = (h,) + outputs[2:]  # add hidden states and attention if they are here
#
#                     loss_fct = CrossEntropyLoss()
#
#                     # carefully choose hyper-parameters
#                     loss = loss_fct(h.view(-1, self.num_labels), labels.view(-1))
#
#                 else:
#                     h += self.fc(dropout(feature))
#                     outputs = (h,) + outputs[2:]  # add hidden states and attention if they are here
#                     loss_fct = CrossEntropyLoss()
#
#                     # carefully choose hyper-parameters
#                     loss = loss_fct(h.view(-1, self.num_labels), labels.view(-1))
#
#             loss = loss / len(self.dropouts)
#
#             outputs = (loss,) + outputs
#         else:
#             for i, dropout in enumerate(self.dropouts):
#                 if i == 0:
#                     h = self.fc(dropout(feature))
#                     outputs = (h,) + outputs[2:]  # add hidden states and attention if they are here
#                 else:
#                     h += self.fc(dropout(feature))
#                     outputs = (h,) + outputs[2:]  # add hidden states and attention if they are here
#
#             outputs = outputs
#         return outputs



class RobertaForSequenceClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        config.output_hidden_states=True

        self.roberta = RobertaModel(config)
        self.fc = nn.Linear(768, self.config.num_labels)

        for param in self.roberta.parameters():
            param.requires_grad = True
        self.weights = nn.Parameter(torch.rand(13, 1))
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
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

        outputs= self.roberta(
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

        feature2 = torch.sum(ht_cls * atten.view(13, 1, 1, 1), dim=[0, 2])

        # if labels is not None:
        #
        #     for i, dropout in enumerate(self.dropouts):
        #         if i == 0:
        #             h = self.fc(dropout(feature))
        #             outputs = (h,) + outputs[2:]  # add hidden states and attention if they are here
        #
        #             h2 = self.fc(dropout(feature2))
        #             loss_fct = CrossEntropyLoss()
        #
        #             # carefully choose hyper-parameters
        #             loss = 0.5*(loss_fct(h.view(-1, self.num_labels), labels.view(-1)) + loss_fct(h2.view(-1, self.num_labels), labels.view(-1)))
        #
        #
        #             kl_loss = compute_kl_loss(feature, feature2)
        #             loss = loss + 4 * kl_loss
        #
        #         else:
        #             h += self.fc(dropout(feature))
        #             outputs = (h,) + outputs[2:]  # add hidden states and attention if they are here
        #             h2 += self.fc(dropout(feature))
        #
        #             loss_fct = CrossEntropyLoss()
        #
        #             # carefully choose hyper-parameters
        #             loss = 0.5*(loss_fct(h.view(-1, self.num_labels), labels.view(-1)) + loss_fct(h2.view(-1, self.num_labels), labels.view(-1)))
        #
        #             kl_loss = compute_kl_loss(feature, feature2)
        #             loss = loss + 4 * kl_loss
        #
        #     loss = loss / len(self.dropouts)
        #
        #     outputs = (loss,) + outputs
        # else:
        #     for i, dropout in enumerate(self.dropouts):
        #         if i == 0:
        #             h = self.fc(dropout(feature))
        #             outputs = (h,) + outputs[2:]  # add hidden states and attention if they are here
        #         else:
        #             h += self.fc(dropout(feature))
        #             outputs = (h,) + outputs[2:]  # add hidden states and attention if they are here
        #
        #     outputs = outputs
        # return outputs

        if labels is not None:

            for i, dropout in enumerate(self.dropouts):
                if i == 0:
                    h = self.fc(dropout(feature))
                    outputs = (h,) + outputs[2:]  # add hidden states and attention if they are here

                    loss_fct = CrossEntropyLoss()

                    # carefully choose hyper-parameters
                    loss = loss_fct(h.view(-1, self.num_labels), labels.view(-1))


                    kl_loss = compute_kl_loss(feature, feature2)
                    loss = loss + 4 * kl_loss

                else:
                    h += self.fc(dropout(feature))
                    outputs = (h,) + outputs[2:]  # add hidden states and attention if they are here
                    loss_fct = CrossEntropyLoss()

                    # carefully choose hyper-parameters
                    loss = loss_fct(h.view(-1, self.num_labels), labels.view(-1))

                    kl_loss = compute_kl_loss(feature, feature2)
                    loss = loss + 4 * kl_loss

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




import torch.nn.functional as F
class RobertaForSequenceClassification_cnn(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        # add cnn params
        config.output_hidden_states = True
        self.filter_sizes_ = (3, 4, 5)  # 卷积核尺寸
        self.num_filters_ = 256  # 卷积核数量(channels数)
        self.hidden_size_ = 768

        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters_, (k, self.hidden_size_)) for k in self.filter_sizes_])

        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear((self.num_filters_ * len(self.filter_sizes_))*3 + self.hidden_size_, config.num_labels)

        self.init_weights()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

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
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import RobertaTokenizer, RobertaForSequenceClassification
        import torch

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

        """
        outputs= self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        # flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        #
        #
        # bert_output_a, pooled_output_a, hidden_output_a = self.roberta(input_ids=flat_input_ids, position_ids=flat_position_ids,
        #                     token_type_ids=flat_token_type_ids,
        #                     attention_mask=flat_attention_mask, head_mask=head_mask)

        # sequence_output = outputs[1]
        # last_cat = torch.cat(
        #     (sequence_output, outputs[2][-1][:,0], outputs[2][-2][:,0], outputs[2][-3][:,0]),
        #     1,
        # )
        sequence_output = outputs[2][-1]
        out1 = sequence_output.unsqueeze(1)
        out1 = torch.cat([self.conv_and_pool(out1, conv) for conv in self.convs], 1)

        sequence_output = outputs[2][-2]
        out2 = sequence_output.unsqueeze(1)
        out2 = torch.cat([self.conv_and_pool(out2, conv) for conv in self.convs], 1)

        sequence_output = outputs[2][-3]
        out3 = sequence_output.unsqueeze(1)
        out3 = torch.cat([self.conv_and_pool(out3, conv) for conv in self.convs], 1)

        out = torch.cat((out1, out2, out3, outputs[1]),1)
        pooled_output = self.dropout(out)
        logits = self.classifier(pooled_output)

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


class xlmroberta_model(RobertaForSequenceClassification):
    """
    This class overrides :class:`~transformers.RobertaForSequenceClassification`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig




class xlmroberta_model_cnn(RobertaForSequenceClassification_cnn):
    """
    This class overrides :class:`~transformers.RobertaForSequenceClassification`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig

















# class bert_cls_model(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         config.output_hidden_states = True
#         self.bert = BertModel(config)
#         for param in self.bert.parameters():
#             param.requires_grad = True
#
#         self.weights = nn.Parameter(torch.rand(13, 1))
#         self.dropouts = nn.ModuleList([
#             nn.Dropout(0.5) for _ in range(5)
#         ])
#         self.fc = nn.Linear(config.hidden_size, self.num_labels)
#
#         self.init_weights()
#
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#     ):
#
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )
#
#         batch_size = input_ids.shape[0]
#         ht_cls = torch.cat(outputs[2])[:, :1, :].view(
#             13, batch_size, 1, 768)
#
#         atten = torch.sum(ht_cls * self.weights.view(
#             13, 1, 1, 1), dim=[1, 3])
#         atten = F.softmax(atten.view(-1), dim=0)
#         feature = torch.sum(ht_cls * atten.view(13, 1, 1, 1), dim=[0, 2])
#
#         if labels is not None:
#
#             for i, dropout in enumerate(self.dropouts):
#                 if i == 0:
#                     h = self.fc(dropout(feature))
#                     outputs = (h,) + outputs[2:]  # add hidden states and attention if they are here
#
#                     loss_fct = CrossEntropyLoss()
#                     loss = loss_fct(h.view(-1, self.num_labels), labels.view(-1))
#                 else:
#                     h += self.fc(dropout(feature))
#                     outputs = (h,) + outputs[2:]  # add hidden states and attention if they are here
#                     loss_fct = CrossEntropyLoss()
#                     loss = loss_fct(h.view(-1, self.num_labels), labels.view(-1))
#
#             loss = loss / len(self.dropouts)
#             outputs = (loss,) + outputs
#         else:
#             for i, dropout in enumerate(self.dropouts):
#                 if i == 0:
#                     h = self.fc(dropout(feature))
#                     outputs = (h,) + outputs[2:]  # add hidden states and attention if they are here
#                 else:
#                     h += self.fc(dropout(feature))
#                     outputs = (h,) + outputs[2:]  # add hidden states and attention if they are here
#
#             outputs = outputs
#
#         return outputs












