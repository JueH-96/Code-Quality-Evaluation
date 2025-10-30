from collections import Counter

N = int(input())
A = [int(x) for x in input().split()]

total = 0
for i in range(N):
    for j in range(i, N):
        total += len(set(A[i:j+1]))

print(total)