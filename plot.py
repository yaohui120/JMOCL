import matplotlib.pyplot as plt
import numpy as np

names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]

# cifar100,每类30样本,精度随稀疏度变化图
x1 = ['20', '40', '60', '80']
y1 = [90.68, 90.48, 86.58, 54.39]
# plt.figure(figsize=(9, 3))
# plt.subplot(133)
# plt.plot(x1, y1)
# plt.xlabel('Sparsity at last task(%)')
# plt.ylabel('Accuracy(%)')
# plt.savefig("acc_sparsity.png",dpi=500, bbox_inches='tight')
# plt.clf()

# cifar100,每任务稀疏5%,精度随保存样本数变化图
x2 = ['1000', '5000', '10000', '15000']
y2 = [88.28, 89.79, 90.97, 91.56]

# imagenetr上，精度随图片质量（样本数量）变化图
x3 = ['95', '85', '75', '55'] # 2000, 3200, 4000, 6000
y3 = [73.55, 74.68, 75.47, 75.37]


# imagenetr上，遗忘率随图片质量（样本数量）变化图
y4 = [7.85, 7.46, 5.97, 6.64]
plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.plot(x3, y3, marker='.')
plt.xlabel('Saved images\' quality(%)')
plt.ylabel('Accuracy(%)')
for a, b in zip(x3, y3):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=8)

plt.subplot(132)
plt.plot(x3, y4, marker='.')
plt.xlabel('Saved images\' quality(%)')
plt.ylabel('Final Forgetting')
for a, b in zip(x3, y4):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=8)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None)
plt.savefig("plot1.png",dpi=1000,bbox_inches = 'tight')
plt.clf()
# \bf{Left}: Last task's accuracy for Split ImagenetR with changing saved images' quality when total sparsity is 30%. The lower the image quality, the more images can be stored.  \bf{Right}: Final Forgetting for Split ImagenetR with changing saved images' quality when total sparsity is 30%. The correspondence between saved images' quality and memory budget is 95%-2000, 85%-3200, 75%-4000, 55%-6000. Under the fixed memory budget, reducing the quality of saved images appropriately to save more samples can better leverage the image's effectiveness.



xx1 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
yy1 = [99.1, 97.85, 97.17, 96.4, 95.16, 94.7, 94.34, 92.81, 92.42, 91.98]
yy2 = [99.2, 98.0, 97.07, 96.42, 95.0, 94.42, 94.16, 91.99, 91.49, 91.3]
yy3 = [99.0, 97.95, 96.97, 96.15, 94.54, 93.68, 93.01, 90.22, 88.13, 85.39]
yy4 = [99.3, 96.1, 95.3, 93.78, 91.76, 90.12, 88.01, 81.14, 70.56, 44.99]
plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.plot(xx1, yy1, label='2% per task', marker='.')
plt.plot(xx1, yy2, label='4% per task', marker='.')
plt.plot(xx1, yy3, label='6% per task', marker='.')
plt.plot(xx1, yy4, label='8% per task', marker='.')
plt.legend()
plt.xlabel('Task')
plt.ylabel('Accuracy(%)')
plt.yticks([40,50,60,70,80,90,100])
# plt.savefig("plot3.png",dpi=1000,bbox_inches = 'tight')
# plt.subplot(132)
# plt.plot(x2, y2, label='5% per task', marker='.')
# plt.xlabel('Memory Budget')
# plt.ylabel('Accuracy(%)')
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None)
# plt.savefig("plot2.png",dpi=1000,bbox_inches = 'tight')
# plt.clf()
# \bf{Left}: Last task's accuracy for Split Cifar100 with changing total sparsity when memory budget is 3000. In fact, when total sparsity is 20%, the sparse space of the model can hold many samples. Here we just save 30 samples per class. When total sparsity exceeds 60%, the model learning ability is greatly affected. \bf{Right}: Last task's accuracy for Split Cifar100 with different memory budget when total spasity is 50%. It can be seen that more saved samples can compensate for the reduced learning ability of the network when the sparsity is not large.



# xx1 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# yy1 = [99.2, 97.75, 96.77, 96.2, 94.56, 93.6, 93.07, 90.72, 89.61, 88.28]
# yy2 = [99.2, 97.95, 97.07, 96.72, 95.16, 94.23, 94.13, 91.95, 90.93, 89.79]
# yy3 = [99.1, 97.7, 97.23, 96.48, 95.3, 94.17, 94.01, 92.18, 91.32, 90.97]
# yy4 = [99.1, 97.75, 97.4, 96.8, 95.7, 94.78, 94.54, 92.52, 92.18, 91.56]
# # plt.figure(figsize=(9, 3))
# plt.subplot(132)
# plt.plot(xx1, yy1, label='MB=1k', marker='.')
# plt.plot(xx1, yy2, label='MB=5k', marker='.')
# plt.plot(xx1, yy3, label='MB=10k', marker='.')
# plt.plot(xx1, yy4, label='MB=15k', marker='.')
# plt.legend()
# plt.xlabel('Task')
# plt.ylabel('Accuracy(%)')
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None)
# plt.savefig("plot2.png",dpi=1000,bbox_inches = 'tight')
# plt.clf()

xx1 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
yy1 = [99.2, 98.1, 97.2, 96.28, 95.52, 94.92, 94.46, 93.14, 92.93, 92.86]
yy2 = [99.4, 97.85, 97.07, 96.45, 95.7, 94.97, 94.59, 93.54, 92.6, 92.52]
yy3 = [99.3, 97.9, 97.1, 96.8, 95.52, 94.75, 94.49, 92.55, 91.99, 91.08]
yy4 = [99.4, 97.6, 97.1, 96.28, 95.26, 93.88, 93.29, 90.58, 86.69, 81.04]
# plt.figure(figsize=(9, 3))
plt.subplot(132)
plt.plot(xx1, yy1, label='2% per task', marker='.')
plt.plot(xx1, yy2, label='4% per task', marker='.')
plt.plot(xx1, yy3, label='6% per task', marker='.')
plt.plot(xx1, yy4, label='8% per task', marker='.')
plt.legend()
plt.xlabel('Task')
plt.ylabel('Accuracy(%)')
plt.yticks([40,50,60,70,80,90,100])
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None)
plt.savefig("plot2.png",dpi=1000,bbox_inches = 'tight')
plt.clf()


# plt.figure(figsize=(9, 3))
# # plt.subplot(131)
# # plt.bar(names, values)
# # plt.subplot(132)
# # plt.scatter(names, values)
# plt.subplot(133)
# plt.plot(x1, y1)
# # plt.title('Categorical Plotting')
# plt.xlabel('Sparsity at last task(%)')
# plt.ylabel('Accuracy(%)')
# plt.savefig("plot.png",dpi=500,bbox_inches = 'tight')
# plt.show()
