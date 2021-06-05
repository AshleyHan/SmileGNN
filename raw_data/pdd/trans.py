def transformt(file_name):
    with open(file_name+'.txt') as infile, open(file_name+'2.txt', 'w') as outfile:
        outfile.write(infile.read().replace("  ", ",\t"))

files = ['approved_example','entity2id','relation2id','train2id']
for file in files:
    #transformt('./'+file)
    print(file)

new_data = []
triple = []
file_path = './train2id.txt'
with open(file_path, encoding='utf8') as reader:
    count = 0
    for line in reader:
        if count == 0:
            count += 1
        else:
            entity1,rel,entity2 = line.strip().split(' ')
            triple = [entity1,entity2,rel]
            new_data.append(triple)

fp = open('train2id2.txt','w')
fp.write(str(len(new_data)))
fp.write('\n')
for triple in new_data:
    fp.write(str(triple[0])+" "+str(triple[1])+" "+str(triple[2]))
    fp.write('\n')
fp.close()