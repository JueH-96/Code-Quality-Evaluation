import math

D = int(input().strip())
x = int(math.sqrt(D))
if x*x == D:
    print(0)
elif x*(x+1) >= D:
    print(1)
else:
    print(2)