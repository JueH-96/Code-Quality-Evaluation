import sys

def solve():
    N = int(input())
    A = [int(x) for x in input().split()]
    
    total = 0
    for i in range(N-1):
        for j in range(i+1, N):
            total += int(max(A[i], A[j]) / min(A[i], A[j]))
    
    return total

print(solve())