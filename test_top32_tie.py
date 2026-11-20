"""Window 87758's measured first two entries; unused fixture entries are synthetic."""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from experiment import TIE_HEAD, TIE_SHA256, digest

ROOT = Path(__file__).resolve().parent
original = next((ROOT / "native/autoresearch").rglob("matrix_heads.py"))
sys.path[:0] = [str(original.parent), str(ROOT/"native/Spec_v2-main")]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

old=load("old_matrix",original)
new=load("tie_matrix",TIE_HEAD)
assert digest(TIE_HEAD)==TIE_SHA256
embedding=nn.Module()
embedding.weight=nn.Parameter(torch.arange(151936,dtype=torch.float32)[:,None].expand(-1,1024),requires_grad=False)
window=dict(round=0,accepted=0,correction=448,rejected=1,offset=1,observation_key="fixture")
record=dict(prompt_ids=[7],output_ids=[448])
rr=dict(observation_key="fixture",accepted_count=0,correction_id=448,bonus_id=None,rejected_id=1,
        prefix_after_commit_length=2,tensor_prefix="x",target_decision_ids=[448],
        top32=dict(used_rows=1,absolute_source_positions=[0],greedy_ids=[448]))
trace=dict(trajectory_policy="RAW",repair_enabled=False,rounds=[rr])
ids=torch.tensor([[714,448,*range(30)]])
scores=torch.tensor([[27.625,27.625,*range(26,-4,-1)]],dtype=torch.bfloat16)
def run(mod, ids, scores):
    return mod.prepare_cached("TOP32_v2",record,window,trace,
        {"x.top32_ids":ids,"x.top32_logits":scores,"x.logsumexp":torch.tensor([28.625])},
        {"draft_sequence":torch.zeros(2,1024)},embedding)

def rejects(fn, ex):
    try: fn()
    except ex: return
    raise AssertionError("bad row accepted")

rejects(lambda:run(old,ids,scores),AssertionError)
with patch.object(new,"semantic_current_top32",wraps=new.semantic_current_top32) as observed:
    out=run(new,ids,scores)
    summary=observed.call_args.args[0]
    assert summary.greedy_token_id==448 and list(summary.top_token_ids)==ids[0].tolist()
    assert list(summary.top_logits_fp16)==scores[0].float().tolist()
ordered=ids.clone()
ordered[0,:2]=torch.tensor([448,714])
assert torch.equal(run(old,ordered,scores).target_top32_features,run(new,ordered,scores).target_top32_features)
lower=scores.clone(); lower[0,1]=27.5
assert torch.equal(run(old,ordered,lower).target_top32_features,run(new,ordered,lower).target_top32_features)
rejects(lambda:run(new,ids,lower),ValueError)
missing=ids.clone();missing[0,1]=447
rejects(lambda:run(new,missing,scores),ValueError)
rr["top32"]["greedy_ids"][0]=714
rejects(lambda:run(new,ids,scores),AssertionError)
print("PASS: old failure; exact tie; saved ordering and values; ordinary parity; lower, missing and wrong RAW greedy rejected")
