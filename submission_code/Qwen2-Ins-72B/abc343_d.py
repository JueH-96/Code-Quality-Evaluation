from bisect import bisect_left, bisect_right

N, T = map(int, input().split())
A = [0] * N
B = []
for _ in range(T):
    a, b = map(int, input().split())
    A[a-1] = max(A[a-1], b)
    B.append((a-1, b))

B.sort(key=lambda x: x[1])
B.append((0, 0))

ans = 1
S = set([0])
for i in range(N):
    if A[i] > 0:
        S.add(A[i])
        ans += 1

for i in range(T):
    if B[i][1] != B[i+1][1]:
        ans = bisect_right(B, (B[i][0], B[i][1])) - bisect_left(B, (B[i][0], 0))
    print(ans)