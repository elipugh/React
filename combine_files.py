import os

folder = 'data/data1'
out = open(folder + "_compiled.csv", "w+")
files = os.listdir('./' + folder + "/")

for line in open(folder + "/" + files[0]):
    out.write(line)

for i in files[1:]:
    file = open(folder + "/" + i)
    file.next()
    for line in file:
        if "status" in line:
            continue
        out.write(line)
    file.close()

out.close()
