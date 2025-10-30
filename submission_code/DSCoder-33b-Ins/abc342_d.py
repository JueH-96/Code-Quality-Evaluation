# YOUR CODE HERE
import sys
import math

def is_square(n):
    return math.isqrt(n) ** 2 == n

def solve():
    N = int(input())
    A = list(map(int, input().split()))
    count = 0
    for i in range(N):
        for j in range(i+1, N):
            if is_square(A[i] * A[j]):
                count += 1
    print(count)

solve()