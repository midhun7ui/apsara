#!/usr/bin/env python3
"""Conservative exact-duplicate/leakage cleaner.
Usage: python clean_apsara_v8_train.py train.jsonl apsara_v8_validation_fixed.jsonl apsara_v8_test_fixed.jsonl
Writes train.cleaned.jsonl and leaves originals unchanged. Exact matching only.
"""
import json,re,sys
from pathlib import Path
def load(p):
 rows=[]
 for n,line in enumerate(Path(p).read_text(encoding="utf-8").splitlines(),1):
  if not line.strip(): continue
  try: r=json.loads(line)
  except json.JSONDecodeError as e: raise SystemExit(f"{p}:{n}: invalid JSON: {e}")
  if not isinstance(r.get("messages"),list): raise SystemExit(f"{p}:{n}: missing messages list")
  rows.append(r)
 return rows
def norm(s): return re.sub(r"\s+"," ",s.casefold()).strip()
def turns(r): return [norm(m.get("content","")) for m in r["messages"] if m.get("role")=="user"]
def main():
 if len(sys.argv)!=4: raise SystemExit("Usage: python clean_apsara_v8_train.py TRAIN.jsonl VALIDATION.jsonl TEST.jsonl")
 tp,vp,xp=sys.argv[1:]; train=load(tp); held=load(vp)+load(xp)
 heldturns={q for r in held for q in turns(r) if q}; seen=set(); kept=[]; dup=[]; leak=[]
 for r in train:
  sig=tuple(turns(r))
  if not sig or sig in seen: dup.append(r.get("id","(no id)")); continue
  if any(q in heldturns for q in sig if q): leak.append(r.get("id","(no id)")); continue
  seen.add(sig); kept.append(r)
 out=Path(tp).with_name("train.cleaned.jsonl")
 out.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in kept),encoding="utf-8")
 print(f"Original train: {len(train)}; cleaned train: {len(kept)}")
 print(f"Removed duplicate conversations: {len(dup)} -> {dup}")
 print(f"Removed train records overlapping validation/test: {len(leak)} -> {leak}")
 print(f"Output: {out}\nNote: exact matching only; manually review semantic duplicates.")
if __name__=="__main__": main()
