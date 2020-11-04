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

        last_layer_hidden = outputs[2][-1]  # [batch_size, seq_len, hidden_dim]
        last_layer_contextual_hidden = outputs[1]  # [batch_size, hidden_dim]

        if labels is not None:
            total_loss = None
            total_weights = 0
            all_logits = []
            # labels_index=labels.detach().cpu().numpy()
            for ix, (label_sample, context_sample, label) in enumerate(
                    zip(last_layer_hidden, list(last_layer_contextual_hidden), labels)):
                label_sample = last_layer_hidden[ix]
                loss_fct = torch.nn.CrossEntropyLoss()

                # label_sample[1]代表第一个label_token对应位置的hidden_states
                # 以此类推
                
                logits_contra = label_sample[1].view(-1) @ context_sample.view(-1, 1)
                logits_neural = label_sample[2].view(-1) @ context_sample.view(-1, 1)
                logits_entail = label_sample[3].view(-1) @ context_sample.view(-1, 1)

                logits = torch.cat([logits_contra, logits_neural, logits_entail], dim=0).unsqueeze(dim=0)

                loss = loss_fct(logits, label.unsqueeze(dim=0))

                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss += loss * (ix + 1)
                total_weights += ix + 1

                all_logits.append(np.squeeze(logits.detach().cpu().numpy()))

        return (total_loss / total_weights,) + (all_logits,) + outputs
```

## Q

依然是随机预测

## baseline

roberta - roberta-base

task: mnli\mnli-mm
evaluation：87.1\86.9
