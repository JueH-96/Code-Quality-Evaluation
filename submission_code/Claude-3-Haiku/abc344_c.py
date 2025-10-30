from itertools import product

# Read input
N = int(input())
A = [int(x) for x in input().split()]
M = int(input())
B = [int(x) for x in input().split()]
L = int(input())
C = [int(x) for x in input().split()]
Q = int(input())
X = [int(x) for x in input().split()]

# Solve the problem
for x in X:
    found = False
    for a, b, c in product(A, B, C):
        if a + b + c == x:
            print("Yes")
            found = True
            break
    if not found:
        print("No")