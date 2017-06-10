# coding=utf-8  
  
import numpy as np  
from numpy.linalg import cholesky  
import matplotlib.pyplot as plt  
  
# 根据不确定参数的分布特性产生随机点
# size:产生随机点的数目
# uncertain_para_list:不确定参数列表
# sigma_list:sigam参数列表

def genRandomPoints(size,uncertain_para_list,sigma_list):

	 
	# 一维正态分布  
	# 下面三种方式是等效的  
	para_num = len(uncertain_para_list)
	if not para_num == len(sigma_list):
		print "the parameter size don't match the sigma size"
		return

	if para_num == 1:
		mu = uncertain_para_list[0]  
		sigma = sigma_list[0]  
		np.random.seed(0)  
		s = np.random.normal(mu, sigma, size )  
		#plt.subplot(1)  
		#plt.hist(s, 30, normed=True)  
	if para_num == 2:	    
		mu1 = uncertain_para_list[0]
		mu2 = uncertain_para_list[1]
		sigma1 = sigma_list[0]
		sigma2 = sigma_list[1]
		mu = np.array([[mu1, mu2]])  
		# 两个参数独立分布
		Sigma = np.array([[sigma1, 0], [0, sigma2]])  
		R = cholesky(Sigma)  
		s = np.dot(np.random.randn(size, 2), R) + mu     
		plt.plot(s[:,0],s[:,1],'+')  
		plt.show() 
	return s

if __name__ == '__main__':
 	uncertain_para_list = [3,4]
 	sigma = [0.5,0.6]
 	ss = genRandomPoints(100,uncertain_para_list,sigma)
 	
