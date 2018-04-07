import scipy.io as sio
import numpy as np
import xml.etree.ElementTree as et
import matplotlib.pyplot as plt
import clearmetrics


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)


xml_colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:orange']
xml_lib = ["GT-1.xml", "GT-2.xml", "GT-3.xml", "GT-4.xml"]
xml_data = {}
xml_data_plot = []

for file in xml_lib:
    tree = et.parse(file)
    root = tree.getroot()
    for frame in root:
        xml_d = []
        for obj in frame:
            for box in obj:
                d = np.array([int(box.attrib['xb']), int(box.attrib['yb'])])
                xml_d.append(d)
                xml_data_plot.append(d)
        xml_data[int(frame.attrib['num'])] = xml_d


xml_data_plot = np.array(xml_data_plot)
ax1.scatter(xml_data_plot[:, 0], xml_data_plot[:, 1], c=xml_colours*750, marker='.')
ax1.set_title('GT_data')

# c=xml_colours*750,


mat_lib = ["data_video_1", "data_video_2", "data_video_3", "data_video_4"]
mat_data = {}
mat_data_plot = []

for file in mat_lib:
    mat_content = sio.loadmat(file+'.mat')[file]
    for cell in mat_content:
        for track in cell:
            for box in track:
                mat_d = np.array([int(box[0]), int(box[1])])
                mat_data_plot.append(mat_d)
                key = int(box[4]) + mat_lib.index(file)*750
                if key in mat_data:
                    mat_data[key].append(mat_d)
                else:
                    mat_data[key] = [mat_d]


mat_data_plot = np.array(mat_data_plot)
ax2.scatter(mat_data_plot[:, 0], mat_data_plot[:, 1], marker='.')
ax2.set_title('Kalman_data')

for i in range(1, 3001):
    if len(xml_data[i]) < 8:
        xml_data[i].append(np.array([1457,  265]))


clear = clearmetrics.ClearMetrics(xml_data, mat_data, 1.5)
clear.match_sequence()
evaluation = [clear.get_mota(),
              clear.get_motp(),
              clear.get_fn_count(),
              clear.get_fp_count(),
              clear.get_mismatches_count(),
              clear.get_object_count(),
              clear.get_matches_count()]

print('MOTA, MOTP, FN, FP, mismatches, objects, matches')
print(evaluation)

plt.show()

