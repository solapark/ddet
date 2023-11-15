import csv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('agg')

def csv2list(file_path, header=True):
    #output : list of list
    f = open(file_path, 'r')
    reader = csv.reader(f)
    my_list = list(reader)
    f.close()
    if not header : my_list = my_list[1:]
    return my_list

fg_dist = csv2list('/data3/sap/Messytable/test_emb/fg.csv', header=False)
bg_dist = csv2list('/data3/sap/Messytable/test_emb/bg.csv', header=False)

fg_dist = np.array(fg_dist).astype('float')
bg_dist = np.array(bg_dist).astype('float')

fg_x, _, fg_obj = plt.hist(fg_dist, color = 'green', alpha = 0.2, bins = 26, range = [0, 1.3], label = 'TP', density = True)
bg_x, _, bg_obj = plt.hist(bg_dist, color = 'red', alpha = 0.2, bins = 26, range = [0, 1.3], label = 'FP', density = True)

for item in fg_obj:
    item.set_height(item.get_height()/sum(fg_x))

for item in bg_obj:
    item.set_height(item.get_height()/sum(bg_x))

plt.legend()

plt.ylim(0,.125)
plt.xlim(0,1.3)

plt.title("Histogram of Embedding Distance")
plt.xlabel("Embedding Distance")
plt.ylabel("Number of Samples (Normalized)")

plt.savefig('/data3/sap/Messytable/test_emb/hist.png')
plt.clf()
