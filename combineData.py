from imp import acquire_lock
from ssl import MemoryBIO
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



old = 0
y = []
indexesTarget = []
index = 0
targetText = []
for line in fileTranslated:
  code = line.split("\t")[0]
  text = line.split("\t")[1]
  if int(code) != int(old):
    targetText.append(" ".join(y).strip())
    indexesTarget.append(index)
    index = index + 1
    y = []

  y.append(text)
  old = code


dfTarget = pd.DataFrame.from_dict({"TargetPT": targetText})

"""
  para gerar inDomainTrainFinal

  dfOrig = dfOrig.drop(0)
  dfOrig = dfOrig.reset_index()
  
  dfOrig = dfOrig.drop(8549)
  dfOrig = dfOrig.reset_index()

  dfTarget = dfTarget.drop(0)
  dfTarget = dfTarget.reset_index()

  para gerar inDomainDevFinal

  dfOrig = dfOrig.drop(0)
  dfOrig = dfOrig.reset_index()

  dfOrig = dfOrig.drop(525)
  dfOrig = dfOrig.reset_index()

  dfTarget = dfTarget.drop(0)
  dfTarget = dfTarget.reset_index()
"""

dfOrig = dfOrig.drop(0)
dfOrig = dfOrig.reset_index()

dfTarget = dfTarget.drop(0)
dfTarget = dfTarget.reset_index()

dfOrig = dfOrig.drop(513)
dfOrig = dfOrig.reset_index()

print(pd.concat([dfTarget, dfOrig], axis=1))




pd.concat([dfTarget, dfOrig], axis=1)[["TargetPT", "annotator", "classe"]].to_parquet('./data/outDomainDevFinal.gzip', compression='gzip')