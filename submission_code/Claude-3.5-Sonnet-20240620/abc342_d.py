# YOUR CODE HERE
from math import isqrt

def is_square(n):
    return isqrt(n)**2 == n

N = int(input())
A = list(map(int, input().split()))

count = 0
square_factors = [set() for _ in range(N)]

for i in range(N):
    if A[i] == 0:
        square_factors[i].add(0)
    else:
        for d in range(1, isqrt(A[i]) + 1):
            if A[i] % d == 0:
                if is_square(d):
                    square_factors[i].add(d)
                if is_square(A[i] // d):
                    square_factors[i].add(A[i] // d)

for i in range(N):
    for j in range(i + 1, N):
        if square_factors[i] & square_factors[j]:
            count += 1

print(count)