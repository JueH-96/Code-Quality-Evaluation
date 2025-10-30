# YOUR CODE HERE
from itertools import product

N = int(input())
A = list(map(int, input().split()))
M = int(input())
B = list(map(int, input().split()))
L = int(input())
C = list(map(int, input().split()))
Q = int(input())
X = list(map(int, input().split()))

for x in X:
    found = False
    for a, b, c in product(A, B, C):
        if a + b + c == x:
            found = True
            break
    print('Yes' if found else 'No')