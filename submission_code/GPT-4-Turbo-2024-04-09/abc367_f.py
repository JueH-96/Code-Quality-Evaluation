import sys
input = sys.stdin.read
from collections import Counter

def solve():
    data = input().split()
    index = 0
    
    N = int(data[index])
    index += 1
    Q = int(data[index])
    index += 1
    
    A = list(map(int, data[index:index+N]))
    index += N
    B = list(map(int, data[index:index+N]))
    index += N
    
    queries = []
    for _ in range(Q):
        l = int(data[index]) - 1
        index += 1
        r = int(data[index]) - 1
        index += 1
        L = int(data[index]) - 1
        index += 1
        R = int(data[index]) - 1
        index += 1
        queries.append((l, r, L, R))
    
    results = []
    for l, r, L, R in queries:
        subA = A[l:r+1]
        subB = B[L:R+1]
        
        if Counter(subA) == Counter(subB):
            results.append("Yes")
        else:
            results.append("No")
    
    print("
".join(results))