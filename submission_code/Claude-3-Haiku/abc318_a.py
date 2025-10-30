import sys

N, M, P = map(int, sys.stdin.readline().split())

count = 0
for day in range(M, N+1, P):
    if day <= N:
        count += 1

print(count)