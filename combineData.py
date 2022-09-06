import pandas as pd

fileSample     = open("./data/outDomainDev.tsv", "r", encoding="utf-8")
fileTranslated = open("./data/outDomainDev.tsv_translated", "r", encoding="utf-8")






y        = []
clase    = []
anotador = []
for line in fileSample:
    cl = line.split("\t")[1]
    an = line.split("\t")[0]
    text = line.split("\t")[3]
    y.append(text.strip())
    clase.append(cl.strip())
    anotador.append(an.strip())

dfOrig = pd.DataFrame.from_dict({"origEN": y, "annotator": anotador, "classe": clase})




y = []
for line in fileTranslated:
  text = line.split("\t")[1]
  y.append(text.strip())
  
  
dfTarget = pd.DataFrame.from_dict({"TargetPT": y})



dfOrig = dfOrig.drop(1)
dfOrig = dfOrig.reset_index()



pd.concat([dfTarget, dfOrig], axis=1).to_parquet('./data/outDomainDevFinal.gzip', compression='gzip')