from collections import defaultdict
import sys
hk = lambda: map(int, sys.stdin.readline().split())

N,*A = hk()
d = defaultdict(int)
d2 = defaultdict(int)

V = int(1e5)
sqrtV = int(V**0.5)+1
candi = [0]*(sqrtV+1)
for i in range(1,sqrtV+1):
    for j in range(i*N,V+1,i*2):
        candi[i] += A[j-1]
    for j in range(sqrtV,i-1,-1):
        candi[i*j//j]**=2
        d[i*j] += candi[i]
        if i != j:
            d[i*j] += candi[j]
    d2[i*i] += sum(A[::i])

print(sum(d.values())+sum(d2.values()))