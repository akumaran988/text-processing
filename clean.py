import re
file=open("F:\\testfile.txt","r")
for line in file:
    if line != "\n":
        line=re.sub(":","",line)
        line=re.sub("-","",line)
        line=re.sub("'","",line)
        line=re.sub(",","",line)
        print(line)
