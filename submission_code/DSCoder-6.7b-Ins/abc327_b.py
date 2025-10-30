import math

def solve(B):
    if B == 1:
        return 1
    for A in range(2, int(math.sqrt(B))+1):
        if A**A == B:
            return A
    return -1

B = int(input())
print(solve(B))