import matplotlib.pyplot as plt

p1 = [[198,860], [222,1729], [1439,1745], [1433, 797]]

x_list = []
y_list = []

for i in range(4):
    x_list.append(p1[i][0])
    y_list.append(p1[i][1])

plt.plot(x_list, y_list, marker="o")

plt.xlim([0, 4025])
plt.ylim([0, 1984])
plt.show()