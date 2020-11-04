# selectedBasedLabelEmbedding

[TOC]

## Loss Function

modeling_roberta.py

line 702 - 775



```python
class RobertaForSelectedBasedLabelModel(RobertaPreTrainedModel):
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=True)

        self.init_weights()

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        contextual_hidden = outputs[2][-1][:, 0, :]  # [batch_size, 1, hidden_dim]
        label_hidden = outputs[2][-1][:, 1:4, :]   # [batch_size, 3, hidden_dim]

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # We are doing regression
                # loss_fct = MSELoss()
                # loss = loss_fct(logits.view(-1), labels.view(-1))
                pass
            else:
                loss_fct = CrossEntropyLoss()
                temp = torch.bmm(label_hidden,
                                 contextual_hidden.view(contextual_hidden.size()[0], -1, 1)).squeeze()  # [batch_size, 3]
                loss = loss_fct(temp,labels.view(-1))

        return (loss,) + (temp,) + outputs
```

## Q

依然是随机预测

## baseline

roberta - roberta-base

task: mnli\mnli-mm
evaluation：87.1\86.9
