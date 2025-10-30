import bisect

N, M, P = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

B.sort()
B_prefix = [0] * (M + 1)
for i in range(M):
    B_prefix[i + 1] = B_prefix[i] + B[i]

total = 0
for a in A:
    target = P - a
    idx = bisect.bisect_right(B, target)
    total += a * idx + B_prefix[idx]
    total += P * (M - idx)

print(total)