from math import gcd
from itertools import accumulate

def solve(A,B,C):
    g = list(accumulate([gcd(a,b) for a,b in zip(A,B)],gcd))
    def f(x,y):
        return sum((a*x+b*y<c) for a,b,c in zip(A,B,C))
    def ok(k):
        if k==0:
            return f(1,1)
        k = -k
        l = 1
        r = 10**18
        while l<r:
            m = (l+r)//2
            if f(k//g[k-1],m)>0:
                r = m
            else:
                l = m+1
        return f(k//g[k-1],l)
    l = -1
    r = 10**18+1
    while l+1!=r:
        m = (l+r)//2
        if ok(m)>0:
            r = m
        else:
            l = m
    return ok(r)

for _ in range(int(input())):
    n = int(input())
    A = []
    B = []
    C = []
    for _ in range(n):
        a,b,c = map(int,input().split())
        A.append(a)
        B.append(b)
        C.append(c)
    print(solve(A,B,C))