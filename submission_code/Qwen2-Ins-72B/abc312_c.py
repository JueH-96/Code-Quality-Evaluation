import sys
input = sys.stdin.readline

N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

A.sort()
B.sort()

A_idx = 0
B_idx = 0

A_count = 0
B_count = 0

ans = 10**9 + 1

while A_idx < N or B_idx < M:
    if A_idx < N and (B_idx == M or A[A_idx] <= B[B_idx]):
        A_count += 1
        if A_count >= B_count + 1:
            ans = min(ans, A[A_idx])
        A_idx += 1
    else:
        B_count += 1
        B_idx += 1

print(ans)