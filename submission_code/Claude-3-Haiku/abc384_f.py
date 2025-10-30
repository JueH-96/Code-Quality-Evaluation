import sys

def f(x):
    while x % 2 == 0:
        x //= 2
    return x

N = int(input())
A = [int(x) for x in input().split()]

total = 0
for i in range(N):
    for j in range(i, N):
        total += f(A[i] + A[j])

print(total)