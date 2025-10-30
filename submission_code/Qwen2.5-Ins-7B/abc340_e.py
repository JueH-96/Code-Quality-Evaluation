# N, M = map(int, input().split())
# A = list(map(int, input().split()))
# B = list(map(int, input().split()))

N, M = 5, 3
A = [1, 2, 3, 4, 5]
B = [2, 4, 0]

cnt = [0] * N
for b in B:
    c = 0
    while True:
        cnt[(b + c) % N] += 1
        c += 1
        if c == 1:
            break

for i in range(N):
    A[i] -= 1
    A[i] += cnt[i]

print(*A)