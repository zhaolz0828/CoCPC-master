import math
import numpy as np
import xlrd
from scipy.integrate import tplquad, dblquad, quad


# 用于计算Gaussian copula的函数值. 输入x, y以及相关系数p, 得出copula值C(F(x), G(y))
def Gaussian_integrate(marginal_x, marginal_y, p):
    Gaussian_copula = []
    for i in range(len(marginal_x)):
        val2, err2 = dblquad(lambda y, x: 1 / (2 * np.pi * math.sqrt(1 - p * p)) * pow(math.e, -(
                    x * x - 2 * p * x * y + y * y) / (2 * (1 - p * p))), float("-inf"), marginal_x[i], float("-inf"),
                             marginal_y[i])
        # print(val2, '\t', marginal_x[i], '\t', marginal_y[i])
        Gaussian_copula.append(val2)
    return Gaussian_copula


# Student-t分布
def Student_t_integrate(marginal_x, marginal_y, p):
    for i in range(len(marginal_x)):
        val2, err2 = dblquad(lambda y, x: 1 / (2 * np.pi * math.sqrt(1 - p * p)) * pow(math.e, 1 + (
                    x * x - 2 * p * x * y + y * y) / (2 * (1 - p * p))), float("-inf"), marginal_x[i], float("-inf"),
                             marginal_y[i])
        print(val2, '\t', marginal_x[i], '\t', marginal_y[i])


# 计算copula的拟合精度
def goodness_of_fit(theory_copula, empirical_copula):
    error = 0
    for i in range(len(theory_copula)):
        error += pow(theory_copula[i] - empirical_copula[i], 2)
    return error


# 计算联合分布函数
def cumulative_function(marginal_x, marginal_y):
    empirical_copula = []
    for i in range(len(marginal_x)):
        count = 0
        for j in range(len(marginal_y)):
            if marginal_x[j] <= marginal_x[i] and marginal_y[j] <= marginal_y[i]:
                count += 1
        empirical_copula.append(count / len(marginal_x))
    return empirical_copula


def Gumbel_copula(marginal_x, marginal_y, cumulative_x, cumulative_y, theta, probability):
    probability_x = []
    probability_y = []
    for i in range(len(marginal_x)):
        for i2 in range(len(cumulative_x)):
            if cumulative_x[i2] == marginal_x[i]:
                probability_x.append(probability[i2])

    for j in range(len(marginal_y)):
        for j2 in range(len(cumulative_y)):
            if cumulative_y[j2] == marginal_y[j]:
                probability_y.append(probability[j2])
    Gumbel_copula = []
    for i in range(len(probability_x)):
        G_copula_x = -math.log(probability_x[i], math.e)
        G_copula_y = -math.log(probability_y[i], math.e)
        # 计算Gumbel_copula的对应值
        Gumbel_copula.append(pow(math.e, -pow((pow(G_copula_x, theta) + pow(G_copula_y, theta)), 1 / theta)))
    return Gumbel_copula


def Clayton_copula(marginal_x, marginal_y, cumulative_x, cumulative_y, theta, probability):
    probability_x = []
    probability_y = []
    # 将marginal_x与对应的分布函数概率统一起来, 方便查找得到u1, u2(对应边际分布函数概率)
    for i in range(len(marginal_x)):
        for i2 in range(len(cumulative_x)):
            if cumulative_x[i2] == marginal_x[i]:
                probability_x.append(probability[i2])

    for j in range(len(marginal_y)):
        for j2 in range(len(cumulative_y)):
            if cumulative_y[j2] == marginal_y[j]:
                probability_y.append(probability[j2])
    Clayton_copula = []
    for i in range(len(probability_x)):
        Clayton_copula.append(pow((pow(probability_x[i], -theta) + pow(probability_y[i], -theta) - 1), -1 / theta))
    return Clayton_copula


if __name__ == "__main__":
    data = xlrd.open_workbook(".\\中国与亚洲国家风险相关性研究.xlsx")
    # data = xlrd.open_workbook(".\\中国股市内部相关性分析.xlsx")
    # data = xlrd.open_workbook(".\\中欧美日-风险相关性研究.xlsx")
    # data = xlrd.open_workbook(".\\中欧美日-风险相关性研究-近3年.xlsx")
    # data = xlrd.open_workbook(".\\中欧美日-风险相关性研究-前3年.xlsx")
    table = data.sheets()[4]
    # 用于计算Gaussian_copula
    print("/*****************************Gaussian_copula*****************************")
    marginal_y = table.col_values(8)
    """
    for k in range(1,8):
        marginal_x = table.col_values(k)
        empirical_copula = cumulative_function(marginal_x, marginal_y)
        for i in range(-9, 10):
            print(i/10, goodness_of_fit(Gaussian_integrate(marginal_x,marginal_y,i/10), empirical_copula))
    for i in range(1, 28):
        marginal_y = table.col_values(i)
        for j in range(i+1, 28):
            marginal_x = table.col_values(i+1)
            empirical_copula = cumulative_function(marginal_x, marginal_y)
            for i in range(-9, 10):
                print(goodness_of_fit(Gaussian_integrate(marginal_x,marginal_y,i/10), empirical_copula), i, j)
    """
    # marginal_x用于表示实际的数据
    marginal_x = table.col_values(1)
    # table_CDF用于查询分布函数概率
    table_CDF = data.sheets()[6]
    cumulative_x = table_CDF.col_values(1)
    cumulative_y = table_CDF.col_values(8)
    probability = table_CDF.col_values(9)

    # 用于计算Gumbel_copula
    print("/*****************************Gumbel_copula*****************************/")
    for j in range(2, 3):
        marginal_x = table.col_values(j)
        cumulative_x = table_CDF.col_values(j)
        for i in range(100, 180):
            print(
                goodness_of_fit(Gumbel_copula(marginal_x, marginal_y, cumulative_x, cumulative_y, i / 100, probability),
                                cumulative_function(marginal_x, marginal_y)), "\t", i / 100)
    """


    #用于计算Clayton_copula
    print("/*****************************Clayton_copula*****************************/")
    for j in range(2, 3):
        marginal_x = table.col_values(j)
        cumulative_x = table_CDF.col_values(j)
        for i in range(1, 200):
            print(goodness_of_fit(Clayton_copula(marginal_x, marginal_y, cumulative_x, cumulative_y, i/100, probability), cumulative_function(marginal_x, marginal_y)), "\t", i/100)
    """
