import os
from utils import data
from subprocess import call

directory = data.FileFinder().get_file('BATS_DIR')
embedding_file = data.FileFinder().get_file('CUSTOM_GLOVE')
print("Reading embedding file for vocabulary...",end='',flush=True)
# Read embedding file for vocabulary
vocab = []
embed_file = open(embedding_file,'r')
for line in embed_file.readlines():
    vocab.append(line.strip().split()[0])
embed_file.close()
print("OK",flush=True)
print("Reading BATS files and merging...",end='',flush=True)
output_file = open(data.FileFinder().get_file('BATS_FULL_FILE'),'w')
# Merge regular and irregular plural files
file1 = os.path.join(directory,"1_Inflectional_morphology/I02 [noun - plural_irreg].txt")
file2 = os.path.join(directory,"1_Inflectional_morphology/I01 [noun - plural_reg].txt")
merged = open(os.path.join(directory,"1_Inflectional_morphology/I01 I02 [noun - plurals].txt"),'w')
file1r = open(file1,'r')
file2r = open(file2,'r')
for line in file1r.readlines():
    merged.write(line)
file1r.close()
for line in file2r.readlines():
    merged.write(line)
file2r.close()
merged.close()
call(["rm","-f",file1])
call(["rm","-f",file2])
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
