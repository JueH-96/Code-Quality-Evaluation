import sys

# Read input
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Combine and sort A and B
C = sorted(A + B)

# Check if any two consecutive elements in C are from A
for i in range(1, len(C)):
    if C[i-1] in A and C[i] in A:
        print("Yes")
        sys.exit()

print("No")