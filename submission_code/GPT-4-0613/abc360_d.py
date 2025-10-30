# YOUR CODE HERE
N, T = map(int, input().split())
S = input()
X = list(map(int, input().split()))

ants = sorted([(x, s == '1') for x, s in zip(X, S)])

cnt = [0] * (N + 1)
for i in range(N - 1, -1, -1):
    cnt[i] = cnt[i + 1] + (ants[i][1] == 0)

res = 0
for i in range(N):
    if ants[i][1] == 1:
        l = i + 1
        r = N
        while r - l > 1:
            m = (l + r) // 2
            if ants[m][0] - ants[i][0] <= T:
                l = m
            else:
                r = m
        res += cnt[l]

print(res)