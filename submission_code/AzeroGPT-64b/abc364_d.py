import math,string,itertools,fractions,heapq,collections,re,array,bisect,sys,random,time

sys.setrecursionlimit(10**7)
inf = 10**20
mod = 10**9 + 7

def main():    
    n,q = map(int, input().split())
    a = list(map(int, input().split()))
    b = []
    k = []
    for i in range(q):
        b0,k0 = map(int, input().split())
        b.append(b0)
        k.append(k0)

    data = []
    for i in range(len(b)):
        diff = []
        for j in range(len(a)):
            d = abs(a[j] - b[i])
            diff.append(d)
        diff.sort()
        data.append(diff[k[i]-1])

    for i in range(len(data)):
        print(data[i])
   
    
main()