from sklearn.metrics import precision_recall_fscore_support
test = []
result = []
all_tags = set()
with open('atis.test.txt','r') as f:
    for line in f.readlines():
        sp = line.strip().split('\t')
        if len(sp) >= 2:    
            words, tags = sp
            tags = tags.split()
            for t in tags:
                test.append(t)

with open('result2.txt','r') as f:
    for line in f.readlines():
        sp = line.strip().split('\t')
        if len(sp) >= 2:    
            words, tags = sp
            tags = tags.split()
            for t in tags:
                all_tags.add(t)
                result.append(t)
all_tags = list(all_tags)
for label in all_tags:
    print label
    print precision_recall_fscore_support(test, result,labels = [label], average='weighted')
print "weighted average"
print precision_recall_fscore_support(test, result, average='weighted')