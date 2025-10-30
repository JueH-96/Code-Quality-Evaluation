from collections import defaultdict

N, T = map(int, input().split())
A = list(map(int, input().split()))

mp = defaultdict(int)
rec = [0] * N
l, r = [0] * N, [0] * N
d1, d2 = 0, 0

for i in range(T):
    a = A[i]
    k = a // N
    j = a % N
    mp[a] += 1
    rec[k] += mp[a]
    l[j] += mp[a]
    if j == k:
        d1 += mp[a]
    if N - k - 1 == j:
        d2 += mp[a]

    if d1 == N or d2 == N or any([rec[k] == N for k in range(N)]) or any([l[j] == N for j in range(N)]):
        print(i + 1)
        exit()

print(-1)