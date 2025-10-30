from bisect import bisect_left

N, Q = map(int, input().split())
A = list(map(int, input().split()))
X = [[] for _ in range(32)]
for i, a in enumerate(A):
    X[a.bit_length()].append(i)

for _ in range(Q):
    i, x = map(int, input().split())
    i -= 1
    a = A[i]
    X[a.bit_length()].pop(bisect_left(X[a.bit_length()], i))
    X[x.bit_length()].append(i)
    A[i] = x

    ans = 0
    while X[ans]:
        if X[ans][0] != ans:
            break
        ans += 1
    print(ans)