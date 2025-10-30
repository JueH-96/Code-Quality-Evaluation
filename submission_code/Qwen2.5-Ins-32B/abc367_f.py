import sys
from collections import Counter

def solve():
    input = sys.stdin.read
    data = input().split()
    
    n, q = int(data[0]), int(data[1])
    A = list(map(int, data[2:n+2]))
    B = list(map(int, data[n+2:2*n+2]))
    
    queries = list(zip(*[iter(map(int, data[2*n+2:]))]*4))
    
    for l, r, L, R in queries:
        if Counter(A[l-1:r]) == Counter(B[L-1:R]):
            print("Yes")
        else:
            print("No")

solve()