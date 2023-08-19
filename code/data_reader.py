import json,os
import random
from tqdm import tqdm
import torch

label_map={"mnli":{0:"Yes",1:"No",2:"Maybe"},
           "mnli_mm":{0:"Yes",1:"No",2:"Maybe"},
           "qnli":{0:"Yes",1:"No"},
           "cb":{0:"Yes",1:"No",2:"Maybe"},
           "wnli":{0:"Yes",1:"No",2:"Maybe"},
           "snli":{0:"Yes",1:"No",2:"Maybe"},
           "rte":{0:"Yes",1:"No"},
           "imdb":{1:"great",0:"terrible"},
           "sst-2":{1:"great",0:"terrible"},
           "cola":{1:"great",0:"terrible"},
           "qqp":{1:"same",0:"different"},
           "mrpc":{1:"same",0:"different"},
           "boolq":{1:"True",0:"False"},
           "multirc":{1:"True",0:"False"},}

def nli_data_set(args,data_set,tokenizer,task_name,if_test=False):
    if(not if_test):
        random.shuffle(data_set)
    index=0
    length=len(data_set)
    

    all_label_text=set()
    for k,v in label_map[task_name].items():
        all_label_text.add(v)
    label_text_map_id=dict()
    for text in all_label_text:
        label_id = tokenizer(' ' + text, add_special_tokens=False)['input_ids']
        assert len(label_id) == 1
        label_text_map_id[text]=label_id[0]
    
    tmp_label_id_list=sorted(list(label_text_map_id.values()))
    label_id_map_index={id:ii for ii,id in enumerate(tmp_label_id_list)}
    label_id_list=tmp_label_id_list
            
    while index < length:
        cur_batch=data_set[index:index+args.batch_size]
        data=['{} ? <mask> , {}'.format(x[0], x[1]) for x in cur_batch]

        label=[ label_text_map_id[label_map[task_name][x[-1]]]  for x in cur_batch]
        label_id_to_index=[label_id_map_index[id] for id in label]
        
        output=tokenizer(data,return_tensors="pt",padding=True,max_length=args.max_length,truncation=True,return_token_type_ids=True)
        
        input_ids=output["input_ids"].numpy().tolist()
        mask_pos_list=[]
        for ii in range(len(input_ids)):
            try:
                cur_mask_pos=input_ids[ii].index(tokenizer.mask_token_id)
            except:
                input_ids[ii][-2] = tokenizer.mask_token_id
                cur_mask_pos=len(input_ids[ii])-2
            mask_pos_list.append(cur_mask_pos)
        output["input_ids"]=torch.tensor(input_ids,dtype=output["input_ids"].dtype)
        output["mask_pos"]=mask_pos_list
        output["label_index"]=label_id_to_index
        output["label_id_lits"]=label_id_list
        
        index+=args.batch_size
        yield output,label

def pair_data_set(args,data_set,tokenizer,task_name,if_test=False):
    if(not if_test):
        random.shuffle(data_set)
    index=0
    length=len(data_set)
    

    all_label_text=set()
    for k,v in label_map[task_name].items():
        all_label_text.add(v)
    label_text_map_id=dict()
    for text in all_label_text:
        label_id = tokenizer(' ' + text, add_special_tokens=False)['input_ids']
        assert len(label_id) == 1
        label_text_map_id[text]=label_id[0]
    
    tmp_label_id_list=sorted(list(label_text_map_id.values()))
    # print(tmp_label_id_list)
    label_id_map_index={id:ii for ii,id in enumerate(tmp_label_id_list)}
    label_id_list=tmp_label_id_list
            
    while index < length:
        cur_batch=data_set[index:index+args.batch_size]
        data=['{} <mask> , {}'.format(x[0], x[1]) for x in cur_batch]

        label=[ label_text_map_id[label_map[task_name][x[-1]]]  for x in cur_batch]
        label_id_to_index=[label_id_map_index[id] for id in label]
        
        output=tokenizer(data,return_tensors="pt",padding=True,max_length=args.max_length,truncation=True,return_token_type_ids=True)
        
        input_ids=output["input_ids"].numpy().tolist()
        mask_pos_list=[]
        for ii in range(len(input_ids)):
            try:
                cur_mask_pos=input_ids[ii].index(tokenizer.mask_token_id)
            except:
                input_ids[ii][-2] = tokenizer.mask_token_id
                cur_mask_pos=len(input_ids[ii])-2
            mask_pos_list.append(cur_mask_pos)
        output["input_ids"]=torch.tensor(input_ids,dtype=output["input_ids"].dtype)
        output["mask_pos"]=mask_pos_list
        output["label_index"]=label_id_to_index
        output["label_id_lits"]=label_id_list
        
        index+=args.batch_size
        yield output,label


def sentiment_data_set(args,data_set,tokenizer,task_name,if_test=False):
    if(not if_test):
        random.shuffle(data_set)
    index=0
    length=len(data_set)
    
    all_label_text=set()
    for k,v in label_map[task_name].items():
        all_label_text.add(v)
    label_text_map_id=dict()
    for text in all_label_text:
        label_id = tokenizer(' ' + text, add_special_tokens=False)['input_ids']
        assert len(label_id) == 1
        label_text_map_id[text]=label_id[0]
    
    tmp_label_id_list=sorted(list(label_text_map_id.values()))
    label_id_map_index={id:ii for ii,id in enumerate(tmp_label_id_list)}
    label_id_list=tmp_label_id_list
        
        
    while index < length:
        cur_batch=data_set[index:index+args.batch_size]
        data=['{} . It was <mask>'.format(x[0]) for x in cur_batch]
        label=[ label_text_map_id[label_map[task_name][x[-1]]]  for x in cur_batch]
        label_id_to_index=[label_id_map_index[id] for id in label]
        
        output=tokenizer(data,return_tensors="pt",padding=True,max_length=args.max_length,truncation=True,return_token_type_ids=True)
        
        input_ids=output["input_ids"].numpy().tolist()
        mask_pos_list=[]
        for ii in range(len(input_ids)):
            try:
                cur_mask_pos=input_ids[ii].index(tokenizer.mask_token_id)
            except:
                input_ids[ii][-2] = tokenizer.mask_token_id
                cur_mask_pos=len(input_ids[ii])-2
            mask_pos_list.append(cur_mask_pos)
        output["input_ids"]=torch.tensor(input_ids,dtype=output["input_ids"].dtype)
        output["mask_pos"]=mask_pos_list
        output["label_index"]=label_id_to_index
        output["label_id_lits"]=label_id_list
        
        
        index+=args.batch_size
        yield output,label

def qa_data_set(args,data_set,tokenizer,task_name,if_test=False):
    if(not if_test):
        random.shuffle(data_set)
    index=0
    length=len(data_set)
    
    all_label_text=set()
    for k,v in label_map[task_name].items():
        all_label_text.add(v)
    label_text_map_id=dict()
    for text in all_label_text:
        label_id = tokenizer(' ' + text, add_special_tokens=False)['input_ids']
        assert len(label_id) == 1
        label_text_map_id[text]=label_id[0]
    
    tmp_label_id_list=sorted(list(label_text_map_id.values()))
    label_id_map_index={id:ii for ii,id in enumerate(tmp_label_id_list)}
    label_id_list=tmp_label_id_list
    
    
    while index < length:
        cur_batch=data_set[index:index+args.batch_size]
        data=['{} ? <mask> , {}, {}'.format(x[0], x[1], x[2]) for x in cur_batch]
        label=[ label_text_map_id[label_map[task_name][x[-1]]]  for x in cur_batch]
        label_id_to_index=[label_id_map_index[id] for id in label]
        
        output=tokenizer(data,return_tensors="pt",padding=True,max_length=args.max_length,truncation=True,return_token_type_ids=True)
        
        input_ids=output["input_ids"].numpy().tolist()
        mask_pos_list=[]
        for ii in range(len(input_ids)):
            try:
                cur_mask_pos=input_ids[ii].index(tokenizer.mask_token_id)
            except:
                input_ids[ii][-2] = tokenizer.mask_token_id
                cur_mask_pos=len(input_ids[ii])-2
            mask_pos_list.append(cur_mask_pos)
        output["input_ids"]=torch.tensor(input_ids,dtype=output["input_ids"].dtype)
        output["mask_pos"]=mask_pos_list
        output["label_index"]=label_id_to_index
        output["label_id_lits"]=label_id_list
        
        index+=args.batch_size
        yield output,label


def read_nli_data(args,task_name,mode):
    out_data=list()
    with open(os.path.join(args.data_dir,task_name,mode)) as in_fp:
        for line in in_fp:
            line=json.loads(line)
            cur_line=[line["sentence_one"],line["sentence_two"],line["label"]]
            out_data.append(cur_line)
    # return out_data[:500]
    return out_data

def read_qa_data(args,task_name,mode):
    out_data=list()
    with open(os.path.join(args.data_dir,task_name,mode)) as in_fp:
        for line in in_fp:
            line=json.loads(line)
            cur_line=[line["passage"],line["question"],line["answer"],line["label"]]
            out_data.append(cur_line)
    # return out_data[:500]
    return out_data

def read_sentiment_data(args,task_name,mode):
    out_data=list()
    with open(os.path.join(args.data_dir,task_name,mode)) as in_fp:
        for line in in_fp:
            line=json.loads(line)
            cur_line=[line["sentence"],line["label"]]
            out_data.append(cur_line)
    # return out_data[:500]
    return out_data

def read_sentence_pair_data(args,task_name,mode):
    out_data=list()
    with open(os.path.join(args.data_dir,task_name,mode)) as in_fp:
        for line in in_fp:
            line=json.loads(line)
            cur_line=[line["sentence_one"],line["sentence_two"],line["label"]]
            if(task_name=="sts-b"):
                cur_line=[line["sentence_one"],line["sentence_two"],line["label"]/5]
            out_data.append(cur_line)
    # return out_data[:500]
    return out_data    

def read_all_train_data_set(args):
    task_name_list=args.specific_task_name
    out_data_generator={}
    for task_name in tqdm(task_name_list,desc="read_data_set"):
        if(task_name in ["mnli","qnli","rte","wnli","cb","snli"]):
            out_data_generator[task_name]=read_nli_data(args,task_name,args.train_file)
        elif(task_name in ["imdb","sst-2","cola"]):
            out_data_generator[task_name]=read_sentiment_data(args,task_name,args.train_file)
        elif(task_name in ["qqp","sts-b","mrpc"]):
            out_data_generator[task_name]=read_sentence_pair_data(args,task_name,args.train_file)
        elif(task_name in ["boolq","multirc"]):
            out_data_generator[task_name]=read_qa_data(args,task_name,args.train_file)
        else:
            raise Exception("data_set generator error: {}".format(task_name))
    return out_data_generator

def read_all_evaluate_data_set(args):
    task_name_list=args.specific_task_name
    out_data_generator={}
    for task_name in tqdm(task_name_list,desc="read_data_set"):
        if(task_name in ["qnli","rte","wnli","cb","snli"]):
            out_data_generator[task_name]=read_nli_data(args,task_name,"dev.jsonl")
        elif(task_name in ["imdb","sst-2","cola"]):
            out_data_generator[task_name]=read_sentiment_data(args,task_name,"dev.jsonl")
        elif(task_name in ["qqp","sts-b","mrpc"]):
            out_data_generator[task_name]=read_sentence_pair_data(args,task_name,"dev.jsonl")
        elif(task_name in ["boolq","multirc"]):
            out_data_generator[task_name]=read_qa_data(args,task_name,"dev.jsonl")
        elif(task_name in ["mnli"]):
            out_data_generator[task_name]=read_nli_data(args,task_name,"dev_matched.jsonl")
            # out_data_generator[task_name+"_mm"]=read_nli_data(args,task_name,"dev_mismatched.jsonl")
        else:
            raise Exception("data_set generator error: {}".format(task_name))
    return out_data_generator

def read_all_test_data_set(args):
    task_name_list=args.specific_task_name
    out_data_generator={}
    for task_name in tqdm(task_name_list,desc="read_data_set"):
        if(task_name in ["qnli","rte","wnli","cb","snli"]):
            out_data_generator[task_name]=read_nli_data(args,task_name,"test.jsonl")
        elif(task_name in ["imdb","sst-2","cola"]):
            out_data_generator[task_name]=read_sentiment_data(args,task_name,"test.jsonl")
        elif(task_name in ["qqp","sts-b","mrpc"]):
            out_data_generator[task_name]=read_sentence_pair_data(args,task_name,"test.jsonl")
        elif(task_name in ["boolq","multirc"]):
            out_data_generator[task_name]=read_qa_data(args,task_name,"test.jsonl")
        elif(task_name in ["mnli"]):
            out_data_generator[task_name]=read_nli_data(args,task_name,"test_matched.jsonl")
            out_data_generator[task_name+"_mm"]=read_nli_data(args,task_name,"test_mismatched.jsonl")
        else:
            raise Exception("data_set generator error: {}".format(task_name))
    return out_data_generator

def gen_train_generator(args,all_data_set,tokenizer,if_test=False):
    out_data_generator={}
    data_minibatch=list()
    for task_name,data_set in tqdm(all_data_set.items()):
        step=len(data_set)//args.batch_size
        if(len(data_set)%args.batch_size !=0):
            step+=1
        data_minibatch.extend([task_name]*step)
        print("{}\t{}".format(task_name,step))
        if(task_name in ["mnli","qnli","rte","wnli","cb","snli","boolq","mnli_mm"]):
            out_data_generator[task_name]=nli_data_set(args,data_set,tokenizer,task_name,if_test)
        elif(task_name in ["imdb","sst-2","cola"]):
            out_data_generator[task_name]=sentiment_data_set(args,data_set,tokenizer,task_name,if_test)
        elif(task_name in ["qqp","sts-b","mrpc"]):
            out_data_generator[task_name]=pair_data_set(args,data_set,tokenizer,task_name,if_test)
        elif(task_name in ["multirc"]):
            out_data_generator[task_name]=qa_data_set(args,data_set,tokenizer,task_name,if_test)
        else:
            raise Exception("data_set generator error: {}".format(task_name))
    random.shuffle(data_minibatch)
    return out_data_generator,data_minibatch
        
