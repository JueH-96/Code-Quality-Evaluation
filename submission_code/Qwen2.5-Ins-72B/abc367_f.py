# YOUR CODE HERE
import sys
from collections import Counter

input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])
A = list(map(int, data[2:N+2]))
B = list(map(int, data[N+2:2*N+2]))
queries = [list(map(int, data[2*N+2+i*4:2*N+6+i*4])) for i in range(Q)]

def can_rearrange(l1, r1, l2, r2):
    return Counter(A[l1-1:r1]) == Counter(B[l2-1:r2])

for query in queries:
    l1, r1, l2, r2 = query
    if can_rearrange(l1, r1, l2, r2):
        print("Yes")
    else:
        print("No")