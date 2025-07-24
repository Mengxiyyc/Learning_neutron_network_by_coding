####这是y2c在学习神经网络设计时自己复现的一些代码，totally handy coded!####

###math_box###
e=2.71828
#1.传输函数Transfer function# According to PAGE-10
#尝试使用单行三元表达式替代，从而压缩空间#_#
def hardlim(n):#硬限制值传输函数
    a=0 if n<0 else 1
    return a
def hardlims(n):#对称硬限制值传输函数
    a=-1 if n<0 else 1
    return a
def purelin(n):#线性传输函数
    a=n
    return a
def satlin(n):#饱和线性传输函数
    a=0 if n<0 else 1 if n>1 else n
    return a
def satlins(n):#饱和线性传输函数
    a=-1 if n<-1 else 1 if n>1 else n
    return a
def poslin(n):#饱和线性传输函数
    a=0 if n<0 else n
    return a
def logsig(n):#对数sigmoid传输函数
    a=1/(1+e**(-n))
    return a
#print(logsig(0))#测试区，传输函数


#2.矩阵计算 不通过其它包，最朴素的从头定义矩阵的计算
test_matrix=[[1,2,3],[4,5,6],[7,8,9]]#以一个3x3矩阵作为示例
example_row=[[1,3,5]]
example_col=[[1],[3],[5]]
#从矩阵中提取第x行或第x列向量，用于后续进行矩阵乘法计算
def row_vector_from_matrix(matrix,row_num):
    return matrix[row_num-1]
#print(row_vector_from_matrix(test_matrix,3))
def column_vector_from_matrix(matrix,col_num):
    vector=[]
    for i in range(0,len(matrix)):
        vector.append(matrix[i][col_num-1])
    return vector
#print(column_vector_from_matrix(test_matrix,2))
#从一个矩阵或向量中获取其[行数，列数]
def dimension(matrix):
    row_num=len(matrix)
    #if type(matrix[0])!=list:
    #    return [1,row_num]
    col_num=len(matrix[0])
    for i in range(0,len(matrix)):
        if len(matrix[i])==col_num:
            continue
        else:
            print("[y2c]dimension模块接受到了一个不正确的矩阵:",matrix)
            return None
    return [row_num,col_num]
#print(dimension([[1],[2],[3],[4]]))
#print(dimension([[1,2,3,4,5]]))
#print(dimension([[1,2,3,4,5,6]]))
#print(dimension(test_matrix))
#定义矩阵加法、减法
def additionAB(matrix1,matrix2):
    if dimension(matrix1)!=dimension(matrix2):
        print("[y2c]addition模块接受到了两个不等规模的矩阵:[矩阵A]",matrix1,"[矩阵B]",matrix2)
        return None
    add_sum=[]
    for i in range(0,len(matrix1)):
        vector=[]
        for j in range(0,len(matrix1[i])):
            a=matrix1[i][j]
            b=matrix2[i][j]
            add=a+b
            vector.append(add)
        add_sum.append(vector)
    return add_sum
def minusAB(matrix1,matrix2):
    if dimension(matrix1)!=dimension(matrix2):
        print("[y2c]minus模块接受到了两个不等规模的矩阵:[矩阵A]",matrix1,"[矩阵B]",matrix2)
        return None
    minus_sum=[]
    for i in range(0,len(matrix1)):
        vector=[]
        for j in range(0,len(matrix1[i])):
            a=matrix1[i][j]
            b=matrix2[i][j]
            minus=a-b
            vector.append(minus)
        minus_sum.append(vector)
    return minus_sum
#print(additionAB(test_matrix,test_matrix))
#print(minusAB(test_matrix,test_matrix))
def dot_vectorAB(vector1,vector2):
    if len(vector1)!=len(vector2):
        print("[y2c]dot_vector模块接受到了两个不等规模的向量:[向量A]",vector1,"[向量B]",vector2)
        return None
    dot=[]
    for i in range (0,len(vector1)):
        dot.append(vector1[i]*vector2[i])
    return(sum(dot))
#print(dot_vectorAB([1,2,3],[4,5,6]))##已验证无误

def dot_productAB(matrix1,matrix2):
    dot_matrix=[]
    for i in range(0,len(matrix1)):
        vec=[]
        row_vec=row_vector_from_matrix(matrix1,i)
        for j in range(0,len(matrix2[0])):
            col_vec=column_vector_from_matrix(matrix2,j+1)#下标问题，直接指定第几行第几列即可，不存在第0行
            #print(row_vec,col_vec)
            vec.append(dot_vectorAB(row_vec,col_vec))
            if dot_vectorAB(row_vec,col_vec) is None:
                print("[y2c]dot_vector模块发生报错，导致dot_prodoct无法输出！")
                return None
        dot_matrix.append(vec)
    return dot_matrix
#print(dot_productAB([[1,2,3]],test_matrix))
#print(dot_productAB([[1,2,3]],[[1],[2],[3]]))
#print(dot_productAB([[1,2,3]],[[1],[2]]))
        
#3.单个神经元        
#单个神经元的净输入Wp+b(net input)
def single_neuron_net_input(w_vector,p_vector,bias):
    if dot_vectorAB(w_vector,p_vector)+bias is None:
        print("[y2c]dot_prodoct模块发生报错，导致神经元无法输出！")
        return None
    else:
        return dot_vectorAB(w_vector,p_vector)+bias

#4.单层神经网络
###请注意输入规范：[]为向量。只有在做矩阵乘法时才需要视作一行或一列矩阵。
def single_layerNN(n,func,w,p,b):
    a=[]#要输出一个n维度向量
    for i in range(0,n):
        w_vec=row_vector_from_matrix(w,i+1)
        bias=b[i]
        net=single_neuron_net_input(w_vec,p,bias)
        a.append(func(net))
    return a
w=[[1,2],[3,-4],[5,6]]
p=[-7,8]
b=[3,6,9]
print(single_layerNN(3,hardlim,w,p,b))
    
