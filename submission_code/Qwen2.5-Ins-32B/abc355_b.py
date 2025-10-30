import sys

def solve():
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    # Combine and sort the sequences
    C = sorted(A + B)
    
    # Check for consecutive elements from A
    for i in range(1, len(C)):
        if C[i] in A and C[i-1] in A and C[i] - C[i-1] == 1:
            print("Yes")
            return
    print("No")

solve()