import os

folder = 'data/comments'
out = open(folder + "_compiled.csv", "a")
files = os.listdir('./' + folder + "/")

for line in open(folder + "/" + files[0]):
    out.write(line)

for i in files[1:]:
    file = open(folder + "/" + i)
    file.next()
    for line in file:
        out.write(line)
    file.close()

out.close()
