from collections import defaultdict

N, Q = map(int, input().split())
X = list(map(int, input().split()))

S = set()
A = [0] * N
cnt = defaultdict(int)

for x in X:
    if x in S:
        S.remove(x)
        for j in range(1, N+1):
            if j in S and cnt[j] > 1:
                A[j-1] -= 1
        cnt[len(S)] -= 1
    else:
        S.add(x)
        cnt[len(S)] += 1
        for j in range(1, N+1):
            if j in S:
                A[j-1] += len(S)

print(*A)