N, K, Q = map(int, input().split())
A = [0] * N

for _ in range(Q):
    x, y = map(int, input().split())
    A[x-1] = y
    sorted_A = sorted(A, reverse=True)
    print(sum(sorted_A[:K]))