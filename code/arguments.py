import argparse,os


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train_shared",action='store_true',help="the train flag, if set True, will train model, if not set, will evaluate in the test data")
    parser.add_argument("--train_cluster",action='store_true',help="the train flag, if set True, will train model, if not set, will evaluate in the test data")
    parser.add_argument("--train_cluster_index",type=str,default=None)
    parser.add_argument("--train_specific",action='store_true')
    parser.add_argument("--train_specific_name",type=str,default=None)
    parser.add_argument("--train_search",action='store_true')
    
    parser.add_argument("--train_file",type=str,default="train.jsonl")
    
    parser.add_argument("--save_model_path",type=str,required=True)
    parser.add_argument("--file_post",type=str,required=True)
    
    parser.add_argument("--device",type=str,default="cuda:0")
    parser.add_argument("--batch_size",type=int,default=16)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1)
    parser.add_argument("--max_length",type=int,default=452)
    parser.add_argument("--warm_up_step_rate",type=float,default=0.1)
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--specific_epoch",type=int,default=100)
    parser.add_argument("--learning_rate",type=float,default=5e-4)
    parser.add_argument("--max_grad_norm",type=float,default=1.0)
    parser.add_argument("--weight_decay",type=float,default=0.1)
    parser.add_argument("--adam_epsilon",type=float,default=1e-8)
    parser.add_argument("--model",type=str,choices=["roberta-large"])
    parser.add_argument("--freeze_backbone",type=bool,default=True)
    parser.add_argument("--prompt_prefix_projection",type=bool,default=False)
    
    parser.add_argument("--shared_layer_start",type=int,default=0)
    parser.add_argument("--shared_layer_end",type=int,default=6)
    parser.add_argument("--cluster_layer_start",type=int,default=6)
    parser.add_argument("--cluster_layer_end",type=int,default=12)
    parser.add_argument("--specific_layer_start",type=int,default=12)
    parser.add_argument("--specific_layer_end",type=int,default=24)
    parser.add_argument("--prompt_length",type=int,default=5)
    parser.add_argument("--project_size",type=int,default=64)
    
    parser.add_argument("--share_shared_prompt",action='store_true')
    parser.add_argument("--share_cluster_prompt",action='store_true')
    parser.add_argument("--share_specific_prompt",action='store_true')
    parser.add_argument("--use_gate",action='store_true')

    parser.add_argument("--abalation_wo_share",action='store_true')
    parser.add_argument("--abalation_wo_cluster",action='store_true')

    parser.add_argument("--atten_add",action='store_true')
    parser.add_argument("--atten_bi",action='store_true')
    parser.add_argument("--atten_trans",action='store_true')
    
    parser.add_argument("--cluster_no_atten",action='store_true')
    
    parser.add_argument("--source_prompt",action='store_true')
    
    parser.add_argument("--cos_schedule",action='store_true')


    # parser.add_argument("--data_dir",type=str,default="/home/dell/chentao/uie_project/data/jsonl_data/100-samples/seed-13")
    parser.add_argument("--data_dir",type=str,default="../data/seed-13")

    args = parser.parse_args()
    
    args.specific_task_name=["mnli","qnli","rte","sst-2","qqp","mrpc"]
    args.cluster_num=4
    args.specific_task_map_cluster={"mnli":0,"qnli":0,"rte":0,"wnli":0,"cb":0,"snli":0,"imdb":1,"sst-2":1,"cola":1,"qqp":2,"mrpc":2,"boolq":3,"multirc":3}
    args.specific_task_map_lr={"mnli":2e-3,"qnli":5e-4,"rte":1e-3,"wnli":5e-3,"cb":5e-3,"snli":2e-5,"imdb":5e-3,"sst-2":1e-3,"qqp":5e-4,"mrpc":1e-3,"boolq":5e-3,"multirc":2e-5,"cola":5e-3}
    args.specific_task_map_label_num={"mnli":3,"qnli":2,"rte":2,"wnli":2,"cb":3,"snli":3,"imdb":2,"sst-2":2,"qqp":2,"sts-b":1,"mrpc":2,"boolq":2,"multirc":2,"cola":2}
    args.save_model_path=os.path.join("../model",args.save_model_path)



    return args    