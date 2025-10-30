import sys

N = int(input())
A = list(map(int, input().split()))

total = 0
for i in range(N):
    for j in range(i+1, N):
        total += max(A[j] - A[i], 0)

print(total)