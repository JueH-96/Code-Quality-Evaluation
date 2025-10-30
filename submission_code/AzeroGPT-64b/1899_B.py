#import math
#import sys
#sys.setrecursionlimit(10**2)
def f(i, k):
    if i + k >r:
        return 0
    else:
        return sum(l[i:i+k])
        
for _ in range(int(input())):
    n=int(input())
    l = list(map(int, input().split()))
    r=n
    m1 =0 
    m2 =0
    for k in range(1, int(n ** 0.5) + 1):
        if n % k == 0:
            m1 = max(m1,abs( f(0, k) -f(k * (n // k - 1), k)))
            p=n//k
            m1 = max(m1,abs( f(0, p) -f(p * (k - 1), p)))
    print(m1)