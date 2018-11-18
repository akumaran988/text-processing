import re
wordDic = {
    "trade": 1,
    "date" : 2 ,
    "input" : 3,
    "isin" : 4,
    "counterparty":5
    }

file=open("F:\\testfile.txt","r")

for line in file:
    if line != "\n":
        for key in wordDic:
            line = line.lower()
            line=re.sub(key,str(wordDic[key]),line)
        print(line)
