# WORKPLACE V4 — two fixed pair-aware configurations, validation-only selection.
import os,json,hashlib,re,gc,time,random,shutil,zipfile,collections,inspect
from pathlib import Path
import numpy as np,pandas as pd,torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer,AutoModelForSequenceClassification,Trainer,TrainingArguments,DataCollatorWithPadding,EarlyStoppingCallback,set_seed
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,classification_report,confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import argparse
parser=argparse.ArgumentParser(description='Reproduce the historical V4 experiment from its exact V3 checkpoint. Requires CUDA.')
parser.add_argument('--init-model',required=True,type=Path,help='Exact V3 contextual checkpoint directory')
parser.add_argument('--output-dir',type=Path,default=Path('runs/workplace_v4'))
cli=parser.parse_args()
assert torch.cuda.is_available(),'This historical FP16 training workflow requires a CUDA GPU; use Colab T4.'
VROOT=cli.output_dir.resolve()
assert not VROOT.exists(),'Choose a new output directory; previous runs are not overwritten.'
VRES=VROOT/'results';VRES.mkdir(parents=True)
DATA=Path(__file__).resolve().parents[1]/'data'/'v4'
v_payload={'parts':{s:json.loads((DATA/(s+'.json')).read_text()) for s in ['train','validation','test']},'audit':json.loads((DATA/'split_audit.json').read_text())}
VLABELS=['NO_COMMITMENT','LATENT_COMMITMENT','QUALIFIED_COMMITMENT','EXPLICIT_COMMITMENT'];v_id2label=dict(enumerate(VLABELS));v_label2id={v:k for k,v in v_id2label.items()}
v_data={s:pd.DataFrame(rs) for s,rs in v_payload['parts'].items()};v_audit=v_payload['audit']
for s,frame in v_data.items():
 (VRES/(s+'.json')).write_text(json.dumps(v_payload['parts'][s],indent=2));frame.to_csv(VRES/(s+'.csv'),index=False)
 assert hashlib.sha256((VRES/(s+'.json')).read_bytes()).hexdigest()==v_audit['sha256'][s]
# Exact overlaps and input provenance checked again in Colab.
def v_norm(x):return re.sub(r'\s+',' ',str(x).lower()).strip()
v_prior={k:set() for k in ['model_input_text','target_utterance','conversation_context']};v_files=[]
for fp in Path('/content').glob('latent_commitment*/**/*.csv'):
 if str(VROOT) in str(fp):continue
 try:
  ff=pd.read_csv(fp)
  if all(k in ff for k in v_prior):
   for k in v_prior:v_prior[k].update(ff[k].dropna().map(v_norm))
   v_files.append(str(fp))
 except Exception:pass
v_audit['runtime_prior_overlap']={s:{k:int(v_data[s][k].map(v_norm).isin(v).sum()) for k,v in v_prior.items()} for s in ['validation','test']}
assert all(v==0 for d in v_audit['runtime_prior_overlap'].values() for v in d.values()),v_audit['runtime_prior_overlap']
v_audit['runtime_prior_files']=v_files
v_audit['metadata_not_serialized']=all(r['model_input_text']==f"[CONTEXT] {r['conversation_context']} [TARGET SPEAKER] employee [TARGET UTTERANCE] {r['target_utterance']}" for rs in v_payload['parts'].values() for r in rs)
assert v_audit['metadata_not_serialized'];(VRES/'split_audit.json').write_text(json.dumps(v_audit,indent=2))
VINIT=cli.init_model.resolve()
assert hashlib.sha256((VINIT/'model.safetensors').read_bytes()).hexdigest()=='a6f8b3fa7565136761183cacbd4d61e9b55e3eb411004368cdaf5130335d5009'
v_tok=AutoTokenizer.from_pretrained(VINIT,use_fast=False)
for s,f in v_data.items():
 lens=list(map(len,v_tok(f.model_input_text.tolist(),truncation=False)['input_ids']));assert max(lens)<=256
 print('FROZEN V4',s,len(f),'counts',f.labels.value_counts().sort_index().to_dict(),'max tokens',max(lens),flush=True)
v_group_ids={g:i for i,g in enumerate(v_data['train'].group_id.unique())}
v_groups=[g.index.tolist() for _,g in v_data['train'].groupby('group_id',sort=False)]
assert all(len(g) in [1,2] for g in v_groups)
# Sampler preserves complete two-row context twins within batches. Singles are randomly paired into units.
class VPairSampler(torch.utils.data.Sampler):
 def __init__(self,groups,seed=20261017):self.groups=groups;self.seed=seed;self.epoch=0
 def set_epoch(self,epoch):self.epoch=epoch
 def __len__(self):return sum(map(len,self.groups))+(sum(len(g)==1 for g in self.groups)%2)
 def __iter__(self):
  rng=random.Random(self.seed+self.epoch);single=[g[0] for g in self.groups if len(g)==1];rng.shuffle(single)
  if len(single)%2:single.append(single[-1])
  units=[list(g) for g in self.groups if len(g)==2]+[single[i:i+2] for i in range(0,len(single),2)]
  rng.shuffle(units)
  for unit in units:yield from unit
v_plain_collator=DataCollatorWithPadding(v_tok,pad_to_multiple_of=8)
def v_collate(features):
 keys=torch.tensor([f['pair_key'] for f in features],dtype=torch.long)
 batch=v_plain_collator([{k:v for k,v in f.items() if k!='pair_key'} for f in features]);batch['pair_key']=keys;return batch
class VPairTrainer(Trainer):
 def _get_train_sampler(self,train_dataset=None):return VPairSampler(v_groups)
 def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
  pair_key=inputs.pop('pair_key');labels=inputs.pop('labels');outputs=model(**inputs);logits=outputs.logits
  loss=F.cross_entropy(logits,labels)
  terms=[]
  for key in pair_key.unique():
   ix=torch.where(pair_key==key)[0]
   if len(ix)==2 and labels[ix[0]]!=labels[ix[1]]:
    a,b=ix[0],ix[1];ya,yb=labels[a],labels[b]
    # Each context should favor its own intended label over the other context's label.
    # Softplus margin penalizes the same class receiving equally high logits in both contexts.
    terms.append((F.softplus(1.0-(logits[a,ya]-logits[b,ya]))+F.softplus(1.0-(logits[b,yb]-logits[a,yb])))/2)
  if terms:loss=loss+.2*torch.stack(terms).mean()
  return (loss,outputs) if return_outputs else loss

def v_metrics(y,p):
 pr,rc,f1,_=precision_recall_fscore_support(y,p,labels=range(4),average='macro',zero_division=0)
 return {'accuracy':float(accuracy_score(y,p)),'macro_precision':float(pr),'macro_recall':float(rc),'macro_f1':float(f1)}
v_valpair=[g.index.to_numpy() for _,g in v_data['validation'][v_data['validation'].bucket.eq('context_pair')].groupby('group_id',sort=False)]
def v_compute(ep):
 pred=ep.predictions.argmax(1);y=ep.label_ids;m=v_metrics(y,pred)
 pairacc=float(np.mean([(pred[ix]==y[ix]).all() for ix in v_valpair]));m['pair_both_correct']=pairacc
 m['selection_score']=.8*m['macro_f1']+.2*pairacc
 return m
v_datasets={}
for s in ['train','validation']:
 f=v_data[s];enc=v_tok(f.model_input_text.tolist(),truncation=True,max_length=256);enc['labels']=f.labels.tolist()
 enc['pair_key']=[v_group_ids[g] if s=='train' else -i-1 for i,g in enumerate(f.group_id)]
 v_datasets[s]=Dataset.from_dict(enc)
# Independently check pair sampler invariants before optimizing any weights.
for ep in [0,1]:
 sampler=VPairSampler(v_groups);sampler.set_epoch(ep);order=list(sampler);seen=collections.Counter(order);assert set(seen)==set(range(len(v_data['train'])))
 for g in v_groups:
  if len(g)==2:
   pos=[order.index(i) for i in g];assert pos[0]//8==pos[1]//8
for _name in list(globals()):
 _obj=globals().get(_name)
 if isinstance(_obj,(Trainer,torch.nn.Module)):globals().pop(_name,None)
_obj=None;gc.collect();torch.cuda.empty_cache()
v_runs={}
for v_name,v_lr in [('pair_lr5e6',5e-6),('pair_lr1e5',1e-5)]:
 set_seed(20261017)
 v_model=AutoModelForSequenceClassification.from_pretrained(VINIT,id2label=v_id2label,label2id=v_label2id)
 v_args=TrainingArguments(output_dir=str(VROOT/(v_name+'_checkpoints')),num_train_epochs=6,learning_rate=v_lr,per_device_train_batch_size=8,per_device_eval_batch_size=16,gradient_accumulation_steps=2,weight_decay=.01,warmup_ratio=.1,fp16=True,optim='adamw_torch',eval_strategy='epoch',save_strategy='epoch',load_best_model_at_end=True,metric_for_best_model='selection_score',greater_is_better=True,save_total_limit=1,save_only_model=True,logging_steps=33,seed=20261017,data_seed=20261017,report_to='none',dataloader_num_workers=0,disable_tqdm=True,remove_unused_columns=False)
 v_trainer=VPairTrainer(model=v_model,args=v_args,train_dataset=v_datasets['train'],eval_dataset=v_datasets['validation'],processing_class=v_tok,data_collator=v_collate,compute_metrics=v_compute,callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
 v_started=time.time();v_trainer.train();v_seconds=time.time()-v_started
 v_dest=VROOT/(v_name+'_model');v_trainer.save_model(v_dest);v_tok.save_pretrained(v_dest)
 v_selected_val=v_trainer.evaluate()
 v_runs[v_name]={'learning_rate':v_lr,'pair_loss_weight':.2,'margin':1.,'best_selection_score':v_trainer.state.best_metric,'selection_score_definition':'.8*validation_macro_f1 + .2*validation_pair_both_correct','epochs':v_trainer.state.epoch,'best_checkpoint':v_trainer.state.best_model_checkpoint,'training_seconds':v_seconds,'validation':v_selected_val,'seed':20261017,'initialization':str(VINIT),'test_used_for_selection':False,'model_sha256':hashlib.sha256((v_dest/'model.safetensors').read_bytes()).hexdigest()}
 pd.DataFrame(v_trainer.state.log_history).to_csv(VRES/(v_name+'_training_log.csv'),index=False)
 (VRES/'training_runs.json').write_text(json.dumps(v_runs,indent=2));print('V4 CANDIDATE COMPLETE',v_name,json.dumps(v_runs[v_name]),flush=True)
 del v_model,v_trainer;gc.collect();torch.cuda.empty_cache()
v_best=max(v_runs,key=lambda k:(v_runs[k]['best_selection_score'],v_runs[k]['validation']['eval_macro_f1'],-v_runs[k]['learning_rate']))
VBEST=VROOT/'best_model';shutil.copytree(VROOT/(v_best+'_model'),VBEST,dirs_exist_ok=True)
v_selection={'selected_candidate':v_best,'selected_model_sha256':v_runs[v_best]['model_sha256'],'selection_used_test':False,'candidate_comparison':v_runs}
(VRES/'selection.json').write_text(json.dumps(v_selection,indent=2));print('V4 MODEL SELECTED — TEST NOT EVALUATED',json.dumps(v_selection,indent=2),flush=True)

# V4 FINAL HOLDOUT — only the validation-selected model; no further model selection below.
assert (VRES/'selection.json').exists() and set(v_runs)=={'pair_lr5e6','pair_lr1e5'}
assert not (VRES/'final_test_logits.npz').exists(),'Test was already evaluated: use cached results.'
v_test=v_data['test'];v_y=v_test.labels.to_numpy();v_pairs=[(g,df.index.to_numpy()) for g,df in v_test[v_test.bucket.eq('context_pair')].groupby('group_id',sort=False)]
def v_probs(logits):
 e=np.exp(logits-logits.max(1,keepdims=True));return e/e.sum(1,keepdims=True)
def v_infer(model,texts):
 arrays=[]
 with torch.inference_mode():
  for i in range(0,len(texts),16):
   enc=v_tok(texts[i:i+16],padding=True,truncation=False,return_tensors='pt').to('cuda');assert enc.input_ids.shape[1]<=256
   arrays.append(model(**enc).logits.float().cpu().numpy())
 return np.concatenate(arrays)
v_summary=[];v_pairmetrics=[];v_logits={};v_predictions={};v_sensitivity=[]
def v_record(name,pred,prob):
 f=v_test.copy();f['prediction_id']=pred;f['predicted_label']=[VLABELS[p] for p in pred];f['confidence']=prob.max(1);f['correct']=pred==v_y
 for k,l in enumerate(VLABELS):f['prob_'+l]=prob[:,k]
 f.to_csv(VRES/(name+'_predictions.csv'),index=False);v_predictions[name]=f
 for subset,mask in [('all',np.ones(len(f),bool)),('everyday',f.bucket.eq('everyday').to_numpy()),('context_pairs',f.bucket.eq('context_pair').to_numpy()),('multi_turn',f.conversation_context.str.count(r'\]:').gt(1).to_numpy())]:
  v_summary.append({'model':name,'subset':subset,'n':int(mask.sum()),'correct':int((pred[mask]==v_y[mask]).sum()),**v_metrics(v_y[mask],pred[mask]),'errors_confidence_ge_0.90':int(((pred[mask]!=v_y[mask])&(prob[mask].max(1)>=.9)).sum())})
 for g,ix in v_pairs:v_pairmetrics.append({'model':name,'group_id':g,'both_correct':bool((pred[ix]==v_y[ix]).all()),'predictions_change':bool(pred[ix[0]]!=pred[ix[1]])})
 pd.DataFrame(classification_report(v_y,pred,labels=range(4),target_names=VLABELS,zero_division=0,output_dict=True)).T.to_csv(VRES/(name+'_classification_report.csv'))
 pd.DataFrame(confusion_matrix(v_y,pred,labels=range(4)),index=VLABELS,columns=VLABELS).to_csv(VRES/(name+'_confusion_matrix.csv'))
 return f
for name,modelpath in [('previous_v3',VINIT),('selected_v4',VBEST)]:
 model=AutoModelForSequenceClassification.from_pretrained(modelpath).to('cuda').eval()
 logits=v_infer(model,v_test.model_input_text.tolist());v_logits[name]=logits;pred=logits.argmax(1);prob=v_probs(logits);f=v_record(name,pred,prob)
 if name=='selected_v4':
  pert={};donorrows=[]
  blank=[f'[CONTEXT]  [TARGET SPEAKER] employee [TARGET UTTERANCE] {t}' for t in v_test.target_utterance]
  pert['blank_context']=v_infer(model,blank)
  for seed in [42,43,44,45,46]:
   rng=np.random.default_rng(seed)
   for attempt in range(10000):
    donor=rng.permutation(len(f))
    if (f.group_id.to_numpy()!=f.group_id.to_numpy()[donor]).all():break
   else:raise RuntimeError('Context derangement failed')
   texts=[f'[CONTEXT] {f.conversation_context.iloc[j]} [TARGET SPEAKER] employee [TARGET UTTERANCE] {t}' for j,t in zip(donor,f.target_utterance)]
   pert['shuffle_'+str(seed)]=v_infer(model,texts)
   donorrows.extend({'seed':seed,'sample_id':f.sample_id.iloc[i],'donor_id':f.sample_id.iloc[j]} for i,j in enumerate(donor))
  for condition,log in pert.items():
   p=log.argmax(1);v_sensitivity.append({'condition':condition,**v_metrics(v_y,p),'prediction_changes':int((p!=pred).sum()),'interpretation':'Sensitivity with original labels; no validated semantic gold for changed context.'})
  np.savez_compressed(VRES/'perturbation_logits.npz',**pert);pd.DataFrame(donorrows).to_csv(VRES/'shuffle_donors.csv',index=False)
 del model;gc.collect();torch.cuda.empty_cache()
np.savez_compressed(VRES/'final_test_logits.npz',labels=v_y,**v_logits)
# Lexical baselines are fixed train-only classifiers; persist per-example outputs and pair scores too.
for col,name in [('target_utterance','tfidf_target_only'),('conversation_context','tfidf_context_only'),('model_input_text','tfidf_full_input')]:
 clf=make_pipeline(TfidfVectorizer(ngram_range=(1,2)),LogisticRegression(max_iter=2000,random_state=20261017))
 clf.fit(v_data['train'][col],v_data['train'].labels);pred=clf.predict(v_test[col]);prob=clf.predict_proba(v_test[col]);v_record(name,pred,prob)
pd.DataFrame(v_summary).to_csv(VRES/'test_metrics.csv',index=False);pd.DataFrame(v_pairmetrics).to_csv(VRES/'context_pair_metrics.csv',index=False);pd.DataFrame(v_sensitivity).to_csv(VRES/'context_sensitivity.csv',index=False)
v_new=v_predictions['selected_v4'];v_old=v_predictions['previous_v3']
v_new[~v_new.correct].to_csv(VRES/'selected_v4_errors.csv',index=False)
v_change=v_new[['sample_id','labels','target_utterance','conversation_context']].copy();v_change['previous_prediction']=v_old.prediction_id;v_change['new_prediction']=v_new.prediction_id;v_change['previous_correct']=v_old.correct;v_change['new_correct']=v_new.correct;v_change.to_csv(VRES/'paired_model_comparison.csv',index=False)
v_info={'initial_model_sha256':'a6f8b3fa7565136761183cacbd4d61e9b55e3eb411004368cdaf5130335d5009','selected_model_sha256':v_selection['selected_model_sha256'],'fresh_test_sha256':v_audit['sha256']['test'],'same_frozen_test_for_all_models':True,'test_used_for_selection':False,'no_model_data_or_label_edits_after_test':True,'test_n':len(v_test),'context_pairs':len(v_pairs),'source':'Agent-authored fictional examples; provisional labels, not independent human gold','limitations':['Small single-author corpus; no independent real-workplace validation','Exact overlap audit is not proof against all semantic overlap or wording artifacts','Old V3 partitions are all training material now; old test scores are not unseen','Perturbed contexts retain original labels only for sensitivity diagnostics']}
(VRES/'evaluation_info.json').write_text(json.dumps(v_info,indent=2))
print('V4 FRESH FINAL TEST\n',pd.DataFrame(v_summary).to_string(index=False),flush=True)
print('CONTEXT TWINS — BOTH CORRECT / PREDICTIONS CHANGE\n',pd.DataFrame(v_pairmetrics).groupby('model')[['both_correct','predictions_change']].sum().to_string(),flush=True)
print('CONTEXT SENSITIVITY\n',pd.DataFrame(v_sensitivity).to_string(index=False),flush=True)
print('No final-test-driven changes or reruns. All96 examples retained.')

print('Selected model:',VBEST,'; results:',VRES)
