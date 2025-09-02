import os, json, pandas as pd

def summarize(root="runs"):
    rows=[]
    if not os.path.exists(root):
        return pd.DataFrame()
    for d in sorted(os.listdir(root)):
        rd=os.path.join(root,d)
        fp=os.path.join(rd,"metrics.csv")
        if os.path.isdir(rd) and os.path.exists(fp):
            try:
                df=pd.read_csv(fp)
                if "round" in df.columns and "val_acc" in df.columns:
                    best=df["val_acc"].max()
                    last=df["val_acc"].iloc[-1]
                    rows.append({"run":d,"best_val_acc":best,"last_val_acc":last,"rows":len(df)})
            except Exception as e:
                rows.append({"run":d,"error":str(e)})
    return pd.DataFrame(rows)

if __name__=="__main__":
    df=summarize()
    if df.empty:
        print("No runs to summarize.")
    else:
        print(df.to_string(index=False))
