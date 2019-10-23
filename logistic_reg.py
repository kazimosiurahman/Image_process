import numpy as np
import matplotlib.pyplot as plt

def draw_line(x1,x2):
    ln=plt.plot(x1,x2)
    plt.pause(0.01)
    ln[0].remove()

def plot_data():
    _,ax=plt.subplots(figsize=(4,4))
    ax.scatter(top_region[:,0],top_region[:,1],color='r')
    ax.scatter(bottom_region[:,0],bottom_region[:,1],color='b')

def sigmoid(score):
    return 1/(1+np.exp(-score))

def calculate_error(line_parameters,points,y):
    p = sigmoid(points*line_parameters)
    return -(1/points.shape[0])*(np.log(p).T *y+ np.log(1-p).T*(1-y))

def gradient_descent(line_parameters,points,y,alpha=0.01):
    m = points.shape[0]
    for i in range(5000):
            p = sigmoid(points*line_parameters)
            gradient = (points.T*(p-y))/m
            line_parameters = line_parameters - alpha*gradient
            w1  = line_parameters.item(0)
            w2  = line_parameters.item(1)
            b  = line_parameters.item(2)
            x1 = np.array([points[:,0].min(),points[:,0].max()])
            #               w1x1 +w2x2+b = 0
            x2 = -b/w2 + x1 * (-w1/w2)
            draw_line(x1,x2)


n_points = 100
random_x1_values = np.random.normal(10,2,n_points)
random_x2_values = np.random.normal(12,2,n_points)
top_region = np.array([random_x1_values,random_x2_values])
random_x1_values

#creating the dataset

np.random.seed(0)
n_points = 100
random_x1_values = np.random.normal(10,2,n_points)
random_x2_values = np.random.normal(12,2,n_points)
random_x1_values_ = np.random.normal(5,2,n_points)
random_x2_values_ = np.random.normal(6,2,n_points)
y_labels = np.array([np.zeros(n_points),np.ones(n_points)]).reshape(2*n_points,1)
bias = np.ones(n_points)
top_region = np.array([random_x1_values,random_x2_values,bias]).T
bottom_region = np.array([random_x1_values_,random_x2_values_,bias]).T
all_points = np.vstack((top_region,bottom_region))
# w1 = -0.2
# w2 = -0.35
# b = 4.5
x1 = np.array([bottom_region[:,0].min(),top_region[:,0].max()])
line_parameters = np.matrix(np.zeros(3)).T
#w1x1 +w2x2+b = 0
# x2 = -b/w2 + x1 * (-w1/w2)
# print(x1,x2)
plot_data()
# draw_line(x1,x2)
gradient_descent(line_parameters,all_points,y_labels,alpha=0.01)