import os

directory = "/home/fcampos/w266_final/data/BATS_3.0"
embedding_file = "/home/fcampos/w266_final/data/glove.6B.300d.txt"
print("Reading embedding file for vocabulary...",end='',flush=True)
# Read embedding file for vocabulary
vocab = []
embed_file = open(embedding_file,'r')
for line in embed_file.readlines():
    vocab.append(line.strip().split()[0])
embed_file.close()
print("OK",flush=True)
print("Reading BATS files and merging...",end='',flush=True)
output_file = open(os.path.join(directory,"full_training_file.txt"),'w')
folders = [folder for folder in os.listdir(directory) if not os.path.isfile(os.path.join(directory,folder))]
for folder in folders:
    current_dir = os.path.join(directory,folder)
    files = [filename for filename in os.listdir(current_dir) if os.path.isfile(os.path.join(current_dir,filename))]
    for filename in files:
        current_file = open(os.path.join(current_dir,filename),"r")
        pairs = []
        for line in current_file.readlines():
            pair = line.strip().split()
            a = pair[0]
            if a in vocab:
                bs = pair[1].strip().split('/')
                for b in bs:
                    if b in vocab:
                        pairs.append((a,b))
        for pair_ab in pairs:
            for pair_cd in pairs:
                if pair_ab[0] != pair_cd[0]:
                    output_file.write("%s %s %s %s\n" % (pair_ab[0],pair_ab[1],pair_cd[0],pair_cd[1]))
        current_file.close()
        print("Finished reading %s" % (filename))
output_file.close()
print("Done!")    
