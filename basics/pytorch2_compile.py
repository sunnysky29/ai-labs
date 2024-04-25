
# 
# pytorch 2.0 新特性，compile 运行，训练


import torch, torchvision
import time
import torch._dynamo as dynamo
import torchvision.models as models

def foo(x,y):
    a = torch.sin(x)
    b = torch.cos(x)
    return a+b    
start = time.time()
compile_model1 = torch.compile(foo)
out =  compile_model1(torch.randn(10,10), torch.randn(10,10) )

end = time.time()
print(f'time1: {end-start}')


start = time.time()
compile_model2 = dynamo.optimize("inductor")(foo)
out =  compile_model2(torch.randn(10,10), torch.randn(10,10) )
end = time.time()
print(f'time2: {end-start}')


start = time.time()
# compile_model = dynamo.optimize("inductor")(foo)
out =  foo(torch.randn(10,10), torch.randn(10,10) )
end = time.time()
print(f'time3: {end-start}')


print(f'测试 训练')

model = models.alexnet()
optmizer = torch.optim.SGD(model.parameters(), lr=0.01)
compile_model = torch.compile(model)

x= torch.randn(16,3,224,224)
optmizer.zero_grad()

start = time.time()
out = compile_model(x)
out.sum().backward()
optmizer.step()
end = time.time()
print(f'time4: {end-start}')



start = time.time()
out = model(x)
out.sum().backward()
optmizer.step()
end = time.time()
print(f'time5: {end-start}')

count = []
for epoch in range(10):
    start = time.time()
    out = compile_model(x)
    out.sum().backward()
    optmizer.step()
    end = time.time()
    count.append(end-start)
    print(f'迭代耗时: {end-start}')
print(sum(count)/len(count))