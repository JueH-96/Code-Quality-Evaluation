# YOUR CODE HERE
from itertools import product
from sys import stdin

N = int(input())
A = list(map(int, input().split()))
M = int(input())
B = list(map(int, input().split()))
L = int(input())
C = list(map(int, input().split()))
Q = int(input())
X = list(map(int, input().split()))

# Generate all possible sums of elements from A, B, and C
possible_sums = set()
for a, b, c in product(A, B, C):
    possible_sums.add(a + b + c)

# Check each X_i
for x in X:
    if x in possible_sums:
        print('Yes')
    else:
        print('No')