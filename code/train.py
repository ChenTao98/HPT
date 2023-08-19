import os
# os.environ["TORCH_HOME"]="/home/dell/chentao/.cache/torch"
# os.environ["HF_HOME"]="/home/dell/chentao/.cache/huggingface"
from arguments import get_args
from model_prompt_ptuning import UniHPTMainModel
from data_reader import gen_train_generator,read_all_train_data_set,read_all_evaluate_data_set,read_all_test_data_set
import torch
from transformers import AdamW, get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
from transformers import BertTokenizer,BertModel,RobertaTokenizer,RobertaModel
import numpy as np
from tqdm import tqdm
import random,time,logging,sys,copy,json
import argparse
import transformers
from utils import compute_metric


transformers.logging.set_verbosity_error()


THC_CACHING_ALLOCATOR=0
logging.basicConfig(level=logging.INFO)

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer():
    def __init__(self,args,total_step,input_model=None):
        if(input_model is None):
            self.model=UniHPTMainModel(args)
        else:
            self.model=input_model
        self.model.to(args.device)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
        {'params': [p for n, p in self.model.named_parameters() if (not any(nd in n for nd in no_decay) and p.requires_grad)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in self.model.named_parameters() if (any(nd in n for nd in no_decay) and p.requires_grad)], 'weight_decay': 0.0}]
        self.bert_optimizer = AdamW(optimizer_grouped_parameters,lr=args.learning_rate,eps=args.adam_epsilon)
        if(args.cos_schedule):
            self.bert_scheduler = get_cosine_schedule_with_warmup(self.bert_optimizer, num_warmup_steps=args.warm_up_step_rate*total_step,
                                                    num_training_steps=total_step)
        else:
            self.bert_scheduler = get_linear_schedule_with_warmup(self.bert_optimizer, num_warmup_steps=args.warm_up_step_rate*total_step,
                                                    num_training_steps=total_step)
        self.best_accuracy=0
    
    def train_one_epoch(self,args,train_data_generator,data_minibatch,cur_epoch,if_evaluate=False,if_train_share=False):
        self.model.train()
        total_loss=0
        count=0
        loop=tqdm(data_minibatch,desc="train {}".format(cur_epoch))
        total_step=0
        # for input_ids,token_type_ids,attention_mask,labels in loop:
        for task_name in loop:
            total_step+=1
            outputs,labels=next(train_data_generator[task_name])
            labels=torch.tensor(labels)
            input_ids,token_type_ids,attention_mask,mask_pos=outputs["input_ids"],outputs["token_type_ids"],outputs["attention_mask"],outputs["mask_pos"]
            if(input_ids.shape[0]>(args.batch_size//args.gradient_accumulation_steps)):
                # self.model.gra
                per_step_sample=args.batch_size//args.gradient_accumulation_steps
                cur_total_step=input_ids.shape[0]//per_step_sample if(input_ids.shape[0]%per_step_sample==0) else input_ids.shape[0]//per_step_sample+1
                for i in range(cur_total_step):
                    logits,loss=self.model(task_name,input_ids[i*per_step_sample:(i+1)*per_step_sample].to(args.device),token_type_ids[i*per_step_sample:(i+1)*per_step_sample].to(args.device),attention_mask[i*per_step_sample:(i+1)*per_step_sample].to(args.device),mask_pos[i*per_step_sample:(i+1)*per_step_sample],labels[i*per_step_sample:(i+1)*per_step_sample].to(args.device),if_evaluate=if_evaluate,if_train_share=if_train_share)
                    loss=loss/args.gradient_accumulation_steps
                    loss.backward()  
                    total_loss+=loss.item()
            else:
                logits,loss=self.model(task_name,input_ids.to(args.device),token_type_ids.to(args.device),attention_mask.to(args.device),mask_pos,labels.to(args.device),if_evaluate=if_evaluate,if_train_share=if_train_share)
                loss.backward()
                total_loss+=loss.item()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
            self.bert_optimizer.step()
            self.bert_scheduler.step()
            self.bert_optimizer.zero_grad()
            # total_loss+=loss.item()
            count+=1
            loop.set_postfix(loss=total_loss/count)
    
    def evaluate(self,args,evaluate_data_set=None):
        self.model.eval()
        if(evaluate_data_set is None):
            evaluate_data_set=read_all_evaluate_data_set(args)
        evaluate_data_generator,data_minibatch=gen_train_generator(args,evaluate_data_set,tokenizer)
        with torch.no_grad():
            for task_name,data_generator in evaluate_data_generator.items():
                predict_all=list()
                labels_all=list()
                if(task_name in ["mnli","mnli_mm"]):
                    origin_name=task_name
                    task_name="mnli"
                for outputs,labels in tqdm(data_generator):
                    labels_all.extend(outputs["label_index"])
                    input_ids,token_type_ids,attention_mask,mask_pos=outputs["input_ids"],outputs["token_type_ids"],outputs["attention_mask"],outputs["mask_pos"]
                    logits=self.model(task_name,input_ids.to(args.device),token_type_ids.to(args.device),attention_mask.to(args.device),mask_pos,if_evaluate=True)
                    if(task_name in ["sts-b"]):
                        predict_cur=logits.cpu().numpy().tolist()
                        predict_all.extend(predict_cur)
                    else:
                        logits=logits[:,outputs["label_id_lits"]]
                        predict_cur=torch.argmax(logits,dim=-1).view(-1).cpu().numpy().tolist()
                        predict_all.extend(predict_cur)
                if(task_name in ["sts-b"]):
                    pass
                elif(task_name in ["multirc"]):
                    pass
                else:
                    
                    result=compute_metric(task_name,np.asarray(predict_all),np.asarray(labels_all))
                    acc=result["acc"]
                    if(task_name=="mnli"):
                        print("{}\t{}".format(origin_name,acc))
                        logging.info("{}\t{}".format(origin_name,acc))
                    else:
                        print("{}\t{}".format(task_name,acc))
                        logging.info("{}\t{}".format(task_name,acc))
                    return result
            
    
    def train(self,args,train_data_generator,data_minibatch,all_data_set,tokenizer,save_model=False,if_train_share=False,if_train_cluster=False):
        evaluate_data_set=read_all_evaluate_data_set(args)
        sst2={"sst-2":evaluate_data_set["sst-2"]}
        evaluate_data_set={"rte":evaluate_data_set["rte"]}
        self.model.eval()
        result=self.evaluate(args,evaluate_data_set)
        result=self.evaluate(args,sst2)
        # sys.exit()
        for epoch in range(args.epochs):
            if(if_train_cluster):
                with open("data_mini_batch_{}.txt".format(args.file_post),"a") as out_fp:
                    out_fp.write(json.dumps(data_minibatch)+"\n")
            self.train_one_epoch(args,train_data_generator,data_minibatch,epoch,if_train_share=if_train_share)
            self.model.eval()
            if(if_train_cluster):
                result=self.evaluate(args,evaluate_data_set)
                result=self.evaluate(args,sst2)
            train_data_generator,data_minibatch=gen_train_generator(args,all_data_set,tokenizer)
            if(save_model):
                torch.save({'state_dict': self.model.state_dict(), 'epoch': epoch}, args.save_model_path+"_epoch_{}.pth".format(epoch))

def get_full_model(args):
    logging.info(args.__dict__)
    tokenizer=RobertaTokenizer.from_pretrained(args.model)
    model=UniHPTMainModel(args)
    return model,tokenizer

def save_share_weight(model,args):
    model.save_share_weight(args)


def load_share_weight(model,args):
    model.load_share_weight(args)

def train_shared_prompt(args,input_model,tokenizer):
    all_data_set=read_all_train_data_set(args)
    train_data_generator,data_minibatch=gen_train_generator(args,all_data_set,tokenizer)
    total_step=len(data_minibatch)*args.epochs
    trainer=Trainer(args,total_step,input_model)
    trainer.train(args,train_data_generator,data_minibatch,all_data_set,tokenizer,if_train_share=True)
    save_share_weight(trainer.model,args)
    return trainer.model


def save_cluster_weight(model,args):
    model.save_cluster_weight(args)

def load_cluster_weight(model,args):
    model.load_cluster_weight(args)

def train_cluster_prompt(args,input_model,tokenizer):
    all_data_set=read_all_train_data_set(args)
    train_data_generator,data_minibatch=gen_train_generator(args,all_data_set,tokenizer)
    total_step=len(data_minibatch)*args.epochs
    trainer=Trainer(args,total_step,input_model)
    trainer.train(args,train_data_generator,data_minibatch,all_data_set,tokenizer,if_train_cluster=True)
    save_cluster_weight(trainer.model,args)
    return trainer.model

def save_specific_weight(model,args,task_name,post_fix=""):
    model.save_specific_weight(args,task_name,post_fix)

def load_specific_weight(model,args,task_name,post_fix=""):
    model.load_specific_weight(args,task_name,post_fix)

def train_specific_prompt(args,task_name,input_model,tokenizer):
    origin_task=args.specific_task_name
    origin_save_model=args.save_model_path
    
    args.specific_task_name=[task_name]
    args.learning_rate=args.specific_task_map_lr[task_name]
    args.save_model_path=args.save_model_path+"_"+task_name
    
    all_data_set=read_all_train_data_set(args)
    train_data_generator,data_minibatch=gen_train_generator(args,all_data_set,tokenizer)
    total_step=len(data_minibatch)*args.epochs
    trainer=Trainer(args,total_step,input_model)
    best_acc=0
    evaluate_data_set=read_all_evaluate_data_set(args)
    best_epoch=0
    best_acc_and_f1=0
    for epoch in range(args.epochs):
        trainer.train_one_epoch(args,train_data_generator,data_minibatch,epoch)
        result=trainer.evaluate(args,evaluate_data_set)
        acc=result["acc"]
        if(acc>best_acc):
            best_acc=acc
            best_epoch=epoch
            save_specific_weight(trainer.model,args,task_name,post_fix="lr_{}_wu_{}".format(args.specific_task_map_lr[task_name],args.warm_up_step_rate))
            logging.info("{}\tbest_acc:{}\tepoch:{}".format(task_name,acc,epoch))
            if(args.train_search):
                with open("result_best_acc_roberta_100sample_multi_search_{}.txt".format(args.file_post),"a") as outfp:
                    outfp.write("{}\tbest_acc:{}\tepoch:{}\n".format(task_name,acc,epoch))
            else:
                with open("result_best_acc_roberta_100sample_multi_{}.txt".format(args.file_post),"a") as outfp:
                    outfp.write("{}\tbest_acc:{}\tepoch:{}\n".format(task_name,acc,epoch))
        if(task_name in ["mrpc","qqp"]):
            if(result["acc_and_f1"]>best_acc_and_f1):
                best_acc_and_f1=result["acc_and_f1"]
                save_specific_weight(trainer.model,args,task_name,post_fix="lr_{}_wu_{}_acc_f1".format(args.specific_task_map_lr[task_name],args.warm_up_step_rate))
                logging.info("{}\tbest_acc_f1:{}\tepoch:{}".format(task_name,best_acc_and_f1,epoch))
                if(args.train_search):
                    with open("result_best_acc_roberta_100sample_multi_search_{}.txt".format(args.file_post),"a") as outfp:
                        outfp.write("{}\tbest_acc_f1:{}\tepoch:{}\n".format(task_name,best_acc_and_f1,epoch))
                else:
                    with open("result_best_acc_roberta_100sample_multi_{}.txt".format(args.file_post),"a") as outfp:
                        outfp.write("{}\tbest_acc_f1:{}\tepoch:{}\n".format(task_name,best_acc_and_f1,epoch))
        train_data_generator,data_minibatch=gen_train_generator(args,all_data_set,tokenizer)
    
    test_data_set=read_all_test_data_set(args)
    load_specific_weight(trainer.model,args,task_name,post_fix="lr_{}_wu_{}".format(args.specific_task_map_lr[task_name],args.warm_up_step_rate))
    result=trainer.evaluate(args,test_data_set)
    if(args.train_search):
        with open("result_best_acc_roberta_100sample_multi_search_test_{}.txt".format(args.file_post),"a") as outfp:
            outfp.write("{}\tbest_acc:{}\tepoch:{}\n".format(task_name,result["acc"],best_epoch))
    else:
        with open("result_best_acc_roberta_100sample_multi_test_{}.txt".format(args.file_post),"a") as outfp:
            outfp.write("{}\tbest_acc:{}\tepoch:{}\n".format(task_name,result["acc"],best_epoch))
    if(task_name in ["mrpc","qqp"]):
        load_specific_weight(trainer.model,args,task_name,post_fix="lr_{}_wu_{}_acc_f1".format(args.specific_task_map_lr[task_name],args.warm_up_step_rate))
        result=trainer.evaluate(args,test_data_set)
        if(args.train_search):
            with open("result_best_acc_roberta_100sample_multi_search_test_{}.txt".format(args.file_post),"a") as outfp:
                outfp.write("{}\tbest_acc_f1:{}\tepoch:{}\n".format(task_name,result["acc_and_f1"],best_epoch))
        else:
            with open("result_best_acc_roberta_100sample_multi_test_{}.txt".format(args.file_post),"a") as outfp:
                outfp.write("{}\tbest_acc_f1:{}\tepoch:{}\n".format(task_name,result["acc_and_f1"],best_epoch))
    args.specific_task_name=origin_task
    args.save_model_path=origin_save_model



if __name__ == '__main__':
    args=get_args()

    
    model,tokenizer=get_full_model(args)
    all_task=args.specific_task_name
    if(args.train_shared):
        model=train_shared_prompt(args,model,tokenizer)
    if(not args.abalation_wo_share):
        load_share_weight(model,args)
        model.freeze_share()
    if(args.train_cluster):
        input_model=copy.deepcopy(model)
        train_cluster_prompt(args,input_model,tokenizer)
    if(not args.abalation_wo_cluster):
        load_cluster_weight(model,args)
        model.freeze_cluster()
    args.epochs=args.specific_epoch
    # if(args.train_specific):
    #     input_model=copy.deepcopy(model)
    #     task_list=args.train_specific_name.split(",")
    #     for task_name in task_list:
    #         train_specific_prompt(args,task_name,input_model,tokenizer)
    if(args.train_specific):
        input_model=copy.deepcopy(model)
        task_list=args.train_specific_name.split(",")
        for task_name in task_list:
            if(args.train_search):
                for lr in [5e-4,1e-3,2e-3,5e-3,1e-2]:
                    for warm_up in [0,0.06,0.1]:
                        with open("result_best_acc_roberta_100sample_multi_search_{}.txt".format(args.file_post),"a") as outfp:
                            outfp.write("learning_rate: {}, warm_up_rate:{}\n".format(lr,warm_up))
                        logging.info("learning_rate: {}, warm_up_rate:{}".format(lr,warm_up))
                        args.specific_task_map_lr[task_name]=lr
                        args.warm_up_step_rate=warm_up
                        train_specific_prompt(args,task_name,input_model,tokenizer)
            else:
                train_specific_prompt(args,task_name,input_model,tokenizer)
            
            
    # main(args,args)