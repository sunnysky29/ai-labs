"""_summary_

online softmax】Flash Attention前传

https://zhuanlan.zhihu.com/p/5078640012

"""



def cal_softmax_l(M):
    """计算sm底数和,∑e^(xi). 按行计算
    Args:
        M : 二维矩阵，[[]]_m*n
    Returns:
        _type_: m*1
    演示：
    M = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]])
    result = cal_softmax_l(M)
            tensor([[ 30.1929],
                [606.4401]])
    """
    exp_M = torch.exp(M)
    # print(f'exp_M: {exp_M}')
    # 计算每行的指数和
    row_sums = exp_M.sum(dim=1, keepdim=True)  # 方向为行 ，保留2维信息
    return row_sums
# M = torch.tensor([[1.0, 2.0, 3.0],
#                   [4.0, 5.0, 6.0]])

# result = cal_softmax_l(M)
# print(result)
    


print(f'========================================1, softmax')
print(
"""
         e^xi
sm(xi) = -----
         ∑e^xi

""")

n = 7
d= 6
torch.manual_seed(456)
x= torch.rand(n,d)
# x = torch.tensor([[-0.3, 0.2, 0.5, 0.7, 0.1, 0.8]])  # 赋特定值

print(f'x: {x}')
# softmax_x = torch.softmax(x, dim=1)
softmax_x = torch.exp(x) / cal_softmax_l(x)
print(f'sm_x: {softmax_x}')

print(f'========================================2, safe-softmax')
print(
"""
             e^(xi-m)
safe-sm(xi) = --------
             ∑e^(xi-m)

""")

row_max = torch.max(x, dim=1).values[:, None]  # [:, None]：将一维张量扩展为二维张量
print(f'row_max: {row_max}')
safe_x = x- row_max
safe_softmax_x = torch.exp(safe_x) / cal_softmax_l(safe_x)
print(f'safe_softmax_x: {safe_softmax_x}')
assert torch.allclose(softmax_x, safe_softmax_x)


print(f'========================================3, online-softmax')
print(
"""
online sm, 递增1个新数据, 已知x1 x2 .... x_n-1 , 新增x_n
                 e^(xi-m_n)
online-sm(xi) = --------
                  l_n
m_n : 最新全部数据最值       
l_n ： 最新全部数据的sm底数和

""")
x_pre = x[:, :-1]  #  1,2...., n-1
x_n = x[:, -1]  #  新增数据
print(f'x: {x} \n  x[-1] : {x_n}  \n , X_pre: {x_pre}')

m_pre = torch.max(x_pre, dim=1).values[:, None]   # （n,1）
l_pre = cal_softmax_l(x_pre - m_pre)  #  # （n,1）
print(f'm_pre: {m_pre},{m_pre.shape} \n l_pre: {l_pre}, {l_pre.shape}')

m_n = torch.max(m_pre, x_n.unsqueeze(1))
l_n = l_pre * torch.exp(m_pre-m_n) + torch.exp(x_n.unsqueeze(1) - m_n)
print(f'm_n: {m_n},{m_n.shape} \n l_pre: {l_n}, {l_n.shape}')
online_softmax_x = torch.exp(x-m_n) / l_n
print(f'online_softmax_x: {online_softmax_x}')
assert torch.allclose(softmax_x, online_softmax_x)


print(f'========================================4, block_online-softmax')
print(
"""
online sm, 递增1个新数据, 已知x1 x2 .... x_n/2 , 新增x_(n/2+1) ... x_n
                                          e^(xi-m_n)
blk_online-sm(xi) = ---------------------------------------------------------
                      l_n=l_pre *(e^(m_pre-m_n)) + l_add *(e^(m_add-m_n))
m_n : 最新全部数据最值       
l_n ： 最新全部数据的sm底数和

""")
x_block = torch.split(x, int(d/2), dim = 1)   # 划分成2组， d/2
print(f'x_block: {x_block} ????')
x_pre =x_block[0]  #  1,2...., n/2
x_add = x_block[1]  #  新增数据, x_(n/2+1) ... x_n

m_pre = torch.max(x_pre, dim=1).values[:, None]   # （n,1）
l_pre = cal_softmax_l(x_pre - m_pre)  #  # （n,1）
print(f'm_pre: {m_pre},{m_pre.shape} \n l_pre: {l_pre}, {l_pre.shape}')

m_add = torch.max(x_add, dim=1).values[:, None]   # （n,1）
l_add = cal_softmax_l(x_add - m_add)  #  # （n,1）


m_n = torch.max(m_pre, m_add)
l_n = l_pre * torch.exp(m_pre-m_n) + l_add * torch.exp(m_add-m_n)
print(f'm_n: {m_n},{m_n.shape} ')
blk_online_softmax_x = torch.exp(x-m_n) / l_n
print(f'blk_online_softmax_x: {blk_online_softmax_x}')
assert torch.allclose(softmax_x, blk_online_softmax_x)
