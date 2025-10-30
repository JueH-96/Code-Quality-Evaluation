import sys

N = int(input())
A = [int(x) for x in input().split()]

total = 0
for i in range(N-1):
    for j in range(i+1, N):
        total += (A[i] + A[j]) % 100000000

print(total)