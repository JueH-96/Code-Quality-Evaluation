import sys

def solve():
    N = int(input())
    X = list(map(int, input().split()))
    
    # Sort the coordinates in ascending order
    X.sort()
    
    # Perform the operations to minimize the sum of coordinates
    for i in range(N-3):
        M = (X[i] + X[i+3]) // 2
        X[i+1] = M - (X[i+1] - X[i])
        X[i+2] = M + (X[i+2] - X[i+3])
    
    # Calculate the minimum sum of coordinates
    return sum(X)

print(solve())