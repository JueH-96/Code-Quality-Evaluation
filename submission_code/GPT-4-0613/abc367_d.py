import sys
from bisect import bisect_right

N, M = map(int, sys.stdin.readline().split())
A = list(map(int, sys.stdin.readline().split()))

cumulative_sum = [0]*(N+1)
for i in range(N):
    cumulative_sum[i+1] = cumulative_sum[i] + A[i]

cumulative_sum_mod = [(a%M, i) for i, a in enumerate(cumulative_sum)]
cumulative_sum_mod.sort()

cnt = 0
j = 0
while j < N+1:
    k = j
    while k < N+1 and cumulative_sum_mod[k][0] == cumulative_sum_mod[j][0]:
        k += 1
    cnt += (k-j)*(k-j-1)//2
    j = k

print(cnt)