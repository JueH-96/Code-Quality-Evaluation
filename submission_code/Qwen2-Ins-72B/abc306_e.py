import heapq

N, K, Q = map(int, input().split())
A = [0] * N
B = []
for _ in range(Q):
    X, Y = map(int, input().split())
    X -= 1
    if A[X] > 0:
        heapq.heappush(B, A[X])
    A[X] = Y
    if len(B) < K:
        continue
    if B[0] < Y:
        heapq.heappop(B)
        heapq.heappush(B, Y)
    elif len(B) > K:
        heapq.heappop(B)
ans = sum(B) + sum(sorted(A)[:K])
print(ans)