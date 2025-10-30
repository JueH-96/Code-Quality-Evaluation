N, M = map(int, input().split())
A = list(map(int, input().split()))
res = [0] * N
j = 0
for i in range(1, N + 1):
    while j < M and A[j] < i:
        j += 1
    if A[j] == i:
        res[i - 1] = 0
    else:
        res[i - 1] = A[j] - i
for val in res:
    print(val)