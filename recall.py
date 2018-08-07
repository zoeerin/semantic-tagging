from sklearn.metrics import precision_recall_fscore_support
f1 = open('restauranttest.bio.txt', 'r')
f2 = open('result.rnn.restaurant', 'r')
test = []
result = []
all_tags = set()
while True:
    line = f1.readline()
    words=line.strip().split()
    if not line:    # indicating end of the file
        break
    if len(words) == 0:
        continue
    test.append(words[0])
    all_tags.add(words[0])


while True:
    line = f2.readline()
    words=line.strip().split()
    if not line:    # indicating end of the file
        break
    if len(words) == 0:
        continue
    result.append(words[0])

all_tags = list(all_tags)
for label in all_tags:
    print('label:' + label)
    ret = precision_recall_fscore_support(test, result,labels = [label], average='weighted')
    print('P={:.4}, R={:.4f}, F={:.4f}'.format(ret[0], ret[1], ret[2]))
    print
print "weighted average"
ret = precision_recall_fscore_support(test, result, average='weighted')
print('P={:.4}, R={:.4f}, F={:.4f}'.format(ret[0], ret[1], ret[2]))
