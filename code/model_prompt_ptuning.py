import torch
import torch.nn as nn
# from transformers import RobertaModel,RobertaForMaskedLM
from transformers.models.bert.modeling_bert import BertEncoder,BertPooler,BertLayer
from transformers.models.roberta.modeling_roberta import (
    RobertaEmbeddings,
    RobertaLayer,
    RobertaLMHead,
    RobertaPreTrainedModel,
    RobertaForCausalLM
)
from transformers.utils import logging
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions,BaseModelOutputWithPastAndCrossAttentions
import sys,json,math
import copy


logger = logging.get_logger(__name__)

class UniHPTHeadLayer(nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        self.specific_head=nn.ModuleDict({task_name:nn.Linear(args.hidden_size,args.specific_task_map_label_num[task_name]) for task_name in args.specific_task_name})
    
    def forward(self,task_name,hidden_state):
        logits=self.specific_head[task_name](hidden_state)
        return logits

class GateLayer(nn.Module):
    def __init__(self,dim,project_dim) -> None:
        super().__init__()
        self.project=nn.Linear(dim*2,project_dim)
        self.gate=nn.Linear(project_dim,dim)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self,prompt_task,prompt_atten):
        prompt_cat=torch.cat((prompt_task,prompt_atten),dim=-1)
        # print(prompt_cat.shape)
        gate_out=self.project(prompt_cat)
        # print(gate_out.shape)
        gate_out=self.sigmoid(self.gate(gate_out))
        # print(gate_out.shape)
        gate_out=torch.mul(prompt_task,gate_out)+torch.mul(prompt_atten,1-gate_out)
        return gate_out

class PrefixEncoder(nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, args):
        super().__init__()
        self.prompt_prefix_projection = args.prompt_prefix_projection
        self.args=args
        self.share_num_layer=args.shared_layer_end-args.shared_layer_start
        self.cluster_num_layer=args.cluster_layer_end-args.cluster_layer_start
        self.specific_num_layer=args.specific_layer_end-args.specific_layer_start
        self.task_num=len(args.specific_task_name)
        self.task_map_index={task_name:ii for ii,task_name in enumerate(args.specific_task_name)}
        self.pooling=nn.AdaptiveAvgPool1d(args.hidden_size)
        self.activate_function=nn.Tanh()
        if(args.share_shared_prompt):
            self.share_prompt=nn.Sequential(nn.Linear(args.hidden_size, args.project_size),
                            self.activate_function,
                            nn.Linear(args.project_size,args.hidden_size*args.prompt_length*2))
        else:
            self.share_prompt=nn.ModuleList(
                [nn.Sequential(nn.Linear(args.hidden_size, args.project_size),
                            self.activate_function,
                            nn.Linear(args.project_size,args.hidden_size*args.prompt_length*2))
                for i in range(self.share_num_layer)])
            
        if(args.share_cluster_prompt):
            self.cluster_prompt=nn.ModuleList([nn.Sequential(
                    nn.Linear(args.hidden_size, args.project_size),
                    self.activate_function,
                    nn.Linear(args.project_size,args.hidden_size*args.prompt_length*2)
                    ) for i in range(self.task_num)])
        else:
            self.cluster_prompt=nn.ModuleList([
                nn.ModuleList([nn.Sequential(
                    nn.Linear(args.hidden_size, args.project_size),
                    self.activate_function,
                    nn.Linear(args.project_size,args.hidden_size*args.prompt_length*2)
                    ) for i in range(self.task_num)]) for i in range(self.cluster_num_layer)
            ])
        if(args.atten_add):

            self.atten_q=nn.Linear(args.hidden_size,args.project_size)
            self.atten_k=nn.Linear(args.hidden_size*args.prompt_length,args.project_size)
            self.atten_v=nn.Linear(args.project_size,1)
            self.layer_norm=nn.LayerNorm(args.hidden_size)

        if(args.atten_trans):
            self.atten_q=nn.Linear(args.hidden_size,args.project_size)
            self.atten_k=nn.Linear(args.hidden_size*args.prompt_length,args.project_size)
            self.atten_v=nn.Linear(args.project_size,args.hidden_size)
            self.layer_norm=nn.LayerNorm(args.hidden_size)
        
        if(args.atten_bi):
            self.atten_bi=nn.Linear(args.hidden_size,args.hidden_size)

        if(args.use_gate):
            self.gate=GateLayer(args.hidden_size,args.project_size*2)
        if(args.share_specific_prompt):
            self.specific_prompt=nn.ModuleDict({task_name:nn.Sequential(
                nn.Linear(args.hidden_size, args.project_size),
                self.activate_function,
                nn.Linear(args.project_size,args.hidden_size*args.prompt_length*2)) for task_name in args.specific_task_name} 
            )
        else:
            self.specific_prompt=nn.ModuleDict({task_name:nn.ModuleList([
                nn.Sequential(nn.Linear(args.hidden_size, args.project_size),
                            self.activate_function,
                            nn.Linear(args.project_size,args.hidden_size*args.prompt_length*2)) for i in range(self.specific_num_layer)
            ]) for task_name in args.specific_task_name} 
            )
        
        # for ii in self.children():
        #     if (isinstance(ii, nn.Linear)):
        #         nn.init.xavier_uniform_(ii.weight)
                # nn.init.xavier_uniform_(ii.weight)
        
    def freeze_share(self):
        for n,p in self.share_prompt.named_parameters():
            p.requires_grad=False
        if(self.prompt_prefix_projection):
            for n,p in self.share_trans.named_parameters():
                p.requires_grad=False
    
    def freeze_cluster(self):
        for n,p in self.cluster_prompt.named_parameters():
            p.requires_grad=False
        # if(self.args.use_gate):
        #     for n,p in self.gate.named_parameters():
        #         p.requires_grad=False
        
        # if(self.args.atten_add or self.args.atten_trans):
        #     for n,p in self.atten_q.named_parameters():
        #         p.requires_grad=False
        #     for n,p in self.atten_k.named_parameters():
        #         p.requires_grad=False
        #     for n,p in self.atten_v.named_parameters():
        #         p.requires_grad=False
        #     for n,p in self.layer_norm.named_parameters():
        #         p.requires_grad=False
        
        # if(self.args.atten_bi):
        #     for n,p in self.atten_bi.named_parameters():
        #         p.requires_grad=False

    def freeze_specific(self,specific_index):
        raise Exception("need Implement")
        # for n,p in self.specific_embedding[specific_index].named_parameters():
        #     p.requires_grad=False
        # if(self.prompt_prefix_projection):
        #     for n,p in self.specific_embedding[specific_index].named_parameters():
        #         p.requires_grad=False
    
    
    def get_share_prompt(self,hidden_state,index):
        if(self.args.share_shared_prompt):
            share_value=self.share_prompt(hidden_state)
        else:
            share_value=self.share_prompt[index](hidden_state)
        return share_value
    
    def cal_attention_score(self,cur_cluster_value,cur_hidden_state,task_name,index=0,if_evaluate=False):
        
        if(self.args.atten_add):
            acti=nn.Tanh()
            # 2*task*project
            wk=self.atten_k(cur_cluster_value)
            wq=self.atten_q(cur_hidden_state.unsqueeze(0))
            wqk=acti(wk+wq)
            attention_score=self.atten_v(wqk).squeeze(-1)
            
        elif(self.args.atten_trans):
            acti=nn.SiLU()
            if(self.args.prompt_length>1):
                pooline_cluster_value=self.pooling(cur_cluster_value)
            else:
                pooline_cluster_value=cur_cluster_value
            
            hdown=self.atten_q(cur_hidden_state)
            hdown=acti(hdown)
            hup=self.atten_v(hdown)
            hout=self.layer_norm(hup.unsqueeze(0)).squeeze(0)
            attention_score=torch.matmul(pooline_cluster_value,hout.unsqueeze(1)).squeeze(-1)
        elif(self.args.atten_bi):
            if(self.args.prompt_length>1):
                pooline_cluster_value=self.pooling(cur_cluster_value)
            else:
                pooline_cluster_value=cur_cluster_value

            pooline_cluster_value=self.atten_bi(pooline_cluster_value)
            attention_score=torch.matmul(pooline_cluster_value,cur_hidden_state.unsqueeze(1)).squeeze(-1)

        else:
            if(self.args.prompt_length>1):
                pooline_cluster_value=self.pooling(cur_cluster_value)
            else:
                pooline_cluster_value=cur_cluster_value
            attention_score=torch.matmul(pooline_cluster_value,cur_hidden_state.unsqueeze(1)).squeeze(-1)
        
        attention_score=attention_score / math.sqrt(self.args.hidden_size)    
        attention_score=nn.Softmax(dim=-1)(attention_score)

        # if(index==(self.cluster_num_layer-1) and if_evaluate):
        #     with open("attention_{}_{}_0.txt".format(task_name,self.args.file_post),"a") as out_fp_0:
        #         with open("attention_{}_{}_1.txt".format(task_name,self.args.file_post),"a") as out_fp_1:
        #             attention_score_0=attention_score[0].cpu().numpy().tolist()
        #             attention_score_1=attention_score[1].cpu().numpy().tolist()
        #             out_fp_0.write(json.dumps(attention_score_0)+"\n")
        #             out_fp_1.write(json.dumps(attention_score_1)+"\n")
        return attention_score


    def get_cluster_prompt(self,hidden_state,index,task_name,if_evaluate=False,if_train_share=False):
        if(self.args.share_cluster_prompt):
            cur_layer_cluster=self.cluster_prompt
        else:
            cur_layer_cluster=self.cluster_prompt[index]
        
        all_batch_task_prompt=[]
        
        for i in range(self.task_num):
            all_batch_task_prompt.append(cur_layer_cluster[i](hidden_state))
        cluster_value=torch.cat(all_batch_task_prompt,dim=1)
        
        
        # cluster_value=self.cluster_embedding(prefix)
        # B*T*Layer*prompt*hidden
        cluster_value=cluster_value.view(-1,self.task_num,self.args.prompt_length,2,self.args.hidden_size).transpose(2,3)
        # cur_task_prompt=cluster_value[:,self.task_map_index[task_name],:]
        batch_prompt=[]
        for ii in range(hidden_state.size(0)):
            # layer * prompt * hidden_size
            cur_task_prompt=cluster_value[ii,self.task_map_index[task_name]]
            if(if_train_share):
                cur_task_prompt=cur_task_prompt.transpose(0,1).reshape(self.args.prompt_length,-1)
                batch_prompt.append(cur_task_prompt.unsqueeze(0))
                continue
            cur_hidden_state=hidden_state[ii]
            # layer * task * hidden
            cur_cluster_value=cluster_value[ii].reshape(self.task_num,2,-1).transpose(0,1)
            # print(cur_cluster_value.shape)
            if(self.args.cluster_no_atten):
                cur_cluster_value=torch.mean(cur_cluster_value,dim=1)
                # print(cur_cluster_value.shape)
                cur_cluster_value=cur_cluster_value.reshape(2,self.args.prompt_length,self.args.hidden_size)
                if(self.args.use_gate):
                    # logger.info("gate")
                    cur_task_prompt=self.gate(cur_task_prompt,cur_cluster_value).transpose(0,1).reshape(self.args.prompt_length,-1)
                else:
                    cur_task_prompt=(cur_task_prompt+cur_cluster_value).transpose(0,1).reshape(self.args.prompt_length,-1)
            else:
                # if(if_train_share):
                #     attention_score=torch.ones((2,self.task_num))
                #     attention_score=nn.Softmax(dim=-1)(attention_score).to(self.args.device)
                # else:
                #     attention_score=self.cal_attention_score(cur_cluster_value,cur_hidden_state,task_name,index,if_evaluate)

                attention_score=self.cal_attention_score(cur_cluster_value,cur_hidden_state,task_name,index,if_evaluate)
                
                
                # if(self.args.prompt_length>1):
                #     pooline_cluster_value=self.pooling(cur_cluster_value)
                # else:
                #     pooline_cluster_value=cur_cluster_value
                # attention_score=torch.matmul(pooline_cluster_value,cur_hidden_state.unsqueeze(1)).squeeze(-1)
                # attention_score=nn.Softmax(dim=-1)(attention_score)
                # layer * hidden
                cur_cluster_value=torch.matmul(attention_score.unsqueeze(1),cur_cluster_value).squeeze(1)
                cur_cluster_value=cur_cluster_value.reshape(2,self.args.prompt_length,self.args.hidden_size)
                if(self.args.use_gate):
                    # logger.info("gate")
                    cur_task_prompt=self.gate(cur_task_prompt,cur_cluster_value).transpose(0,1).reshape(self.args.prompt_length,-1)
                else:
                    cur_task_prompt=(cur_task_prompt+cur_cluster_value).transpose(0,1).reshape(self.args.prompt_length,-1)
            batch_prompt.append(cur_task_prompt.unsqueeze(0))
        cluster_value=torch.cat(batch_prompt,dim=0)
        return cluster_value
    
    def get_specific_prompt(self,hidden_state,index,task_name):
        if(self.args.share_specific_prompt):
            specific_value=self.specific_prompt[task_name](hidden_state)
        else:
            specific_value=self.specific_prompt[task_name][index](hidden_state)
        return specific_value

    def forward(self,task_name, prefix: torch.Tensor,hidden_state):
        if self.prompt_prefix_projection:
            share_tokens = self.share_embedding(prefix)
            share_value=self.share_trans(share_tokens)
            cluster_tokens=self.cluster_embedding[self.args.specific_task_map_cluster[task_name]](prefix)
            cluster_value=self.cluster_trans[self.args.specific_task_map_cluster[task_name]](cluster_tokens)
            specific_tokens=self.specific_embedding[task_name](prefix)
            specific_value=self.specific_trans[task_name](specific_tokens)
        else:
            
            cluster_value=self.cluster_embedding(prefix)
            # B*T*Layer*prompt*hidden
            cluster_value=cluster_value.view(-1,self.args.prompt_length,self.task_num,self.cluster_num_layer*2,self.args.hidden_size).transpose(1, 2).transpose(2,3)
            # cur_task_prompt=cluster_value[:,self.task_map_index[task_name],:]
            batch_prompt=[]
            for ii in range(hidden_state.size(0)):
                # layer * prompt * hidden_size
                cur_task_prompt=cluster_value[ii,self.task_map_index[task_name]]
                cur_hidden_state=hidden_state[ii]
                # layer * task * hidden
                cur_cluster_value=cluster_value[ii].view(self.task_num,self.cluster_num_layer*2,-1).transpose(0,1)
                pooline_cluster_value=self.pooling(cur_cluster_value)
                attention_score=torch.matmul(pooline_cluster_value,cur_hidden_state.unsqueeze(1)).squeeze(-1)
                attention_score=nn.Softmax(dim=-1)(attention_score)
                # layer * hidden
                cur_cluster_value=torch.matmul(attention_score.unsqueeze(1),cur_cluster_value).squeeze(1)
                cur_cluster_value=cur_cluster_value.view(self.cluster_num_layer*2,self.args.prompt_length,self.args.hidden_size)
                cur_task_prompt=(cur_task_prompt+cur_cluster_value).transpose(0,1).view(self.args.prompt_length,-1)
                batch_prompt.append(cur_task_prompt.unsqueeze(0))
            cluster_value=torch.cat(batch_prompt,dim=0)
            
            specific_value=self.specific_embedding[task_name](prefix)
        
        
        past_key_values=torch.cat((share_value,cluster_value),dim=2)
        past_key_values=torch.cat((past_key_values,specific_value),dim=2)
        return past_key_values



def define_loss_fct():
    return nn.ModuleDict({"mnli":nn.CrossEntropyLoss(),
                          "qnli":nn.CrossEntropyLoss(),
                          "rte":nn.CrossEntropyLoss(),
                          "wnli":nn.CrossEntropyLoss(),
                          "cb":nn.CrossEntropyLoss(),
                          "snli":nn.CrossEntropyLoss(),
                          "imdb":nn.CrossEntropyLoss(),
                          "sst-2":nn.CrossEntropyLoss(),
                          "qqp":nn.CrossEntropyLoss(),
                          "sts-b":nn.MSELoss(),
                          "mrpc":nn.CrossEntropyLoss(),
                          "boolq":nn.CrossEntropyLoss(),
                          "multirc":nn.CrossEntropyLoss()})

class RobertaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def get_prompt(self, task_name, batch_size,type,index,hidden_state=None,if_evaluate=False,if_train_share=False):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.args.device)
        if(type=="share"):
            past_key_values = self.prefix_encoder.get_share_prompt(hidden_state,index)
        elif(type=="cluster"):
            past_key_values=self.prefix_encoder.get_cluster_prompt(hidden_state,index,task_name,if_evaluate,if_train_share=if_train_share)
        elif(type=="specific"):
            past_key_values=self.prefix_encoder.get_specific_prompt(hidden_state,index,task_name)
        else:
            raise Exception("prompt_type wrong")
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def add_prompt_layer(self,args,config):
        self.pre_seq_len=args.prompt_length
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads
        
        self.prefix_encoder=PrefixEncoder(args)
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.args=args

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        task_name=None,
        if_evaluate=False,
        if_train_share=False
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        
        batch_size=hidden_states.shape[0]
        i_sub_value=0
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            if(i==self.args.shared_layer_start):
                type="share"
                
            elif(i==self.args.cluster_layer_start):
                type="cluster"
                i_sub_value=self.args.cluster_layer_start
                # past_key_values=self.get_prompt(task_name,batch_size,"cluster",i-i_sub_value,hidden_states[:,0,:])
                
            elif(i==self.args.specific_layer_start):
                type="specific"
                i_sub_value=self.args.specific_layer_start
                # past_key_values=self.get_prompt(task_name,batch_size,"specific",i-i_sub_value,hidden_states[:,0,:])
            past_key_values=self.get_prompt(task_name,batch_size,type,i-i_sub_value,hidden_states[:,0,:],if_evaluate,if_train_share=if_train_share)
            # i=i-i_sub_value
            past_key_value = past_key_values[0] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )




class RobertaModel(RobertaPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need`_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.

    .. _`Attention is all you need`: https://arxiv.org/abs/1706.03762

    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)

        # self.pooler = RobertaPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    
    def add_prompt_layer(self,args,config):
        self.encoder.add_prompt_layer(args,config)
        if(args.source_prompt):
            self.source_prompt=nn.Embedding(len(args.specific_task_name),config.hidden_size)
            self.task_map_id={x:ii for ii,x in enumerate(args.specific_task_name)}


    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        args=None,
        task_name=None,
        if_evaluate=False,
        if_train_share=False
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        # past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        past_key_values_length=past_key_values
        

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        if(args.source_prompt):
            source_prompt_id=torch.full((embedding_output.shape[0],1),self.task_map_id[task_name]).to(args.device)
            source_prompt_emb=self.source_prompt(source_prompt_id)
            embedding_output=torch.cat((embedding_output[:,0:1,:],source_prompt_emb,embedding_output[:,1:,:]),dim=1)
            
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            task_name=task_name,
            if_evaluate=if_evaluate,
            if_train_share=if_train_share
        )
        sequence_output = encoder_outputs[0]
        pooled_output=None
        # pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )




class UniHPTMainModel(nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        
        self.backbone=RobertaModel.from_pretrained(args.model)
        config=self.backbone.config
        self.config=config
        if(args.freeze_backbone):
            for n,p in self.backbone.named_parameters():
                p.requires_grad=False
        tmp_model=RobertaForCausalLM.from_pretrained(args.model)
        self.lm_head=tmp_model.lm_head
        if(args.freeze_backbone):
            for n,p in self.backbone.named_parameters():
                p.requires_grad=False
            for n,p in self.lm_head.named_parameters():
                p.requires_grad=False
        
        
        args.hidden_size=config.hidden_size
        args.prefix_hidden_size=config.hidden_size
        self.args=args
        self.backbone.add_prompt_layer(args,config)
        self.pre_seq_len=args.prompt_length

        
        
        # self.n_layer = config.num_hidden_layers
        # self.n_head = config.num_attention_heads
        # self.n_embd = config.hidden_size // config.num_attention_heads
        
        # self.prefix_encoder=PrefixEncoder(args)
        # self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        # self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        total=0
        for n,p in self.backbone.named_parameters():
            if(p.requires_grad):
                total+=p.numel()
        print(total)
        print("length:{}\tper:{}".format(args.prompt_length,total/len(args.specific_task_name)))

        # self.head_layer=UniHPTHeadLayer(args)
        
        self.loss_fct=define_loss_fct()
        self.sigmoid=nn.Sigmoid()
        self.mse=nn.MSELoss()
        self.cross=nn.CrossEntropyLoss()
        
    def save_share_weight(self,args):
        torch.save(self.backbone.encoder.prefix_encoder.state_dict(), args.save_model_path+"share_prefixencoder.pth")
        # torch.save(self.head_layer.state_dict(), args.save_model_path+"share_head_layer.pth")
        if(args.source_prompt):
            torch.save(self.backbone.source_prompt.state_dict(),args.save_model_path+"share_source_prompt.pth")
    
    def load_share_weight(self,args):
        self.backbone.encoder.prefix_encoder.load_state_dict(torch.load(args.save_model_path+"share_prefixencoder.pth"))
        # self.head_layer.load_state_dict(torch.load(args.save_model_path+"share_head_layer.pth"))
        if(args.source_prompt):
            self.backbone.source_prompt.load_state_dict(torch.load(args.save_model_path+"share_source_prompt.pth"))
    
    def save_cluster_weight(self,args):
        torch.save(self.backbone.encoder.prefix_encoder.cluster_prompt.state_dict(),args.save_model_path+"cluster.pth")
        if(args.source_prompt):
            torch.save(self.backbone.source_prompt.state_dict(),args.save_model_path+"cluster_source_prompt.pth")
        if(args.use_gate):
            torch.save(self.backbone.encoder.prefix_encoder.gate.state_dict(),args.save_model_path+"gate.pth")

        if(args.atten_add or args.atten_trans):
            torch.save(self.backbone.encoder.prefix_encoder.atten_q.state_dict(),args.save_model_path+"atten_add_q.pth")
            torch.save(self.backbone.encoder.prefix_encoder.atten_k.state_dict(),args.save_model_path+"atten_add_k.pth")
            torch.save(self.backbone.encoder.prefix_encoder.atten_v.state_dict(),args.save_model_path+"atten_add_v.pth")
            torch.save(self.backbone.encoder.prefix_encoder.layer_norm.state_dict(),args.save_model_path+"atten_layer_norm.pth")
        
        if(args.atten_bi):
            torch.save(self.backbone.encoder.prefix_encoder.atten_bi.state_dict(),args.save_model_path+"atten_add_bi.pth")
    
    def load_cluster_weight(self,args):
        self.backbone.encoder.prefix_encoder.cluster_prompt.load_state_dict(torch.load(args.save_model_path+"cluster.pth"))
        if(args.source_prompt):
            self.backbone.source_prompt.load_state_dict(torch.load(args.save_model_path+"cluster_source_prompt.pth"))
        if(args.use_gate):
            self.backbone.encoder.prefix_encoder.gate.load_state_dict(torch.load(args.save_model_path+"gate.pth"))
        
        if(args.atten_add or args.atten_trans):
            self.backbone.encoder.prefix_encoder.atten_q.load_state_dict(torch.load(args.save_model_path+"atten_add_q.pth"))
            self.backbone.encoder.prefix_encoder.atten_k.load_state_dict(torch.load(args.save_model_path+"atten_add_k.pth"))
            self.backbone.encoder.prefix_encoder.atten_v.load_state_dict(torch.load(args.save_model_path+"atten_add_v.pth"))
            self.backbone.encoder.prefix_encoder.layer_norm.load_state_dict(torch.load(args.save_model_path+"atten_layer_norm.pth"))
        
        if(args.atten_bi):
            self.backbone.encoder.prefix_encoder.atten_bi.load_state_dict(torch.load(args.save_model_path+"atten_add_bi.pth"))
    
    def save_specific_weight(self,args,task_name,post_fix=""):
        torch.save(self.backbone.encoder.prefix_encoder.specific_prompt[task_name].state_dict(),args.save_model_path+"specific_{}.pth".format(task_name+post_fix))
        # torch.save(self.head_layer.specific_head[task_name].state_dict(),args.save_model_path+"specific_head_layer_{}.pth".format(task_name+post_fix))
        if(args.source_prompt):
            torch.save(self.backbone.source_prompt.state_dict(),args.save_model_path+"specific_source_prompt_{}.pth".format(task_name+post_fix))
        if(args.use_gate):
            torch.save(self.backbone.encoder.prefix_encoder.gate.state_dict(),args.save_model_path+"gate_{}.pth".format(task_name+post_fix))

        if(args.atten_add or args.atten_trans):
            torch.save(self.backbone.encoder.prefix_encoder.atten_q.state_dict(),args.save_model_path+"atten_add_q_{}.pth".format(task_name+post_fix))
            torch.save(self.backbone.encoder.prefix_encoder.atten_k.state_dict(),args.save_model_path+"atten_add_k_{}.pth".format(task_name+post_fix))
            torch.save(self.backbone.encoder.prefix_encoder.atten_v.state_dict(),args.save_model_path+"atten_add_v_{}.pth".format(task_name+post_fix))
            torch.save(self.backbone.encoder.prefix_encoder.layer_norm.state_dict(),args.save_model_path+"atten_layer_norm_{}.pth".format(task_name+post_fix))
        
        if(args.atten_bi):
            torch.save(self.backbone.encoder.prefix_encoder.atten_bi.state_dict(),args.save_model_path+"atten_add_bi_{}.pth".format(task_name+post_fix))
    
    def load_specific_weight(self,args,task_name,post_fix=""):
        self.backbone.encoder.prefix_encoder.specific_prompt[task_name].load_state_dict(torch.load(args.save_model_path+"specific_{}.pth".format(task_name+post_fix)))
        # self.head_layer.specific_head[task_name].load_state_dict(torch.load(args.save_model_path+"specific_head_layer_{}.pth".format(task_name+post_fix)))
        if(args.source_prompt):
            self.backbone.source_prompt.load_state_dict(torch.load(args.save_model_path+"specific_source_prompt_{}.pth".format(task_name+post_fix)))
        if(args.use_gate):
            self.backbone.encoder.prefix_encoder.gate.load_state_dict(torch.load(args.save_model_path+"gate_{}.pth".format(task_name+post_fix)))
        
        if(args.atten_add or args.atten_trans):
            self.backbone.encoder.prefix_encoder.atten_q.load_state_dict(torch.load(args.save_model_path+"atten_add_q_{}.pth".format(task_name+post_fix)))
            self.backbone.encoder.prefix_encoder.atten_k.load_state_dict(torch.load(args.save_model_path+"atten_add_k_{}.pth".format(task_name+post_fix)))
            self.backbone.encoder.prefix_encoder.atten_v.load_state_dict(torch.load(args.save_model_path+"atten_add_v_{}.pth".format(task_name+post_fix)))
            self.backbone.encoder.prefix_encoder.layer_norm.load_state_dict(torch.load(args.save_model_path+"atten_layer_norm_{}.pth".format(task_name+post_fix)))
        
        if(args.atten_bi):
            self.backbone.encoder.prefix_encoder.atten_bi.load_state_dict(torch.load(args.save_model_path+"atten_add_bi_{}.pth".format(task_name+post_fix)))
    

    def freeze_share(self):
        self.backbone.encoder.prefix_encoder.freeze_share()
    
    def freeze_cluster(self):
        self.backbone.encoder.prefix_encoder.freeze_cluster()
    
    def freeze_specific(self):
        self.backbone.encoder.prefix_encoder.freeze_specific()

    def forward(self,task_name,input_ids=None,token_type_ids=None,attention_mask=None,mask_pos=None,label=None,if_evaluate=False,if_train_share=False):
        
        batch_size=input_ids.shape[0]
        
        past_key_values=self.pre_seq_len
        if(self.args.source_prompt):
            past_key_values+=1
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len+1).to(self.args.device)
        else:
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.args.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        
        hidden_state=self.backbone(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,past_key_values=past_key_values,args=self.args,task_name=task_name,if_evaluate=if_evaluate,if_train_share=if_train_share)
        hidden_state=hidden_state[0]
        sequence_mask_output = hidden_state[torch.arange(hidden_state.size(0)), mask_pos]
        prediction_scores = self.lm_head(sequence_mask_output)
        
        masked_lm_loss = None
        if label is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), label.view(-1))
        
        # hidden_state=hidden_state[0]
        # hidden_state=hidden_state[1]
        # logits=self.head_layer(task_name,hidden_state)
        # if(task_name in ["sts-b"]):
        #     logits=self.sigmoid(logits).view(-1)
        # if(label is not None):
        #     loss=self.loss_fct[task_name](logits,label)
        # if(task_name not in ["sts-b"]):
        #     logits=torch.softmax(logits,dim=-1)
        # if(label is not None):
        #     return logits,loss
        if(label is not None):
            
            return prediction_scores,masked_lm_loss
        else:
            return prediction_scores
    