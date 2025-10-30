from collections import defaultdict

def mex(A):
    i = 0
    while i in A:
        i += 1
    return i

N, Q = map(int, input().split())
A = list(map(int, input().split()))

count = defaultdict(int)
for a in A:
    count[a] += 1

for _ in range(Q):
    i, x = map(int, input().split())
    i -= 1
    count[A[i]] -= 1
    if count[A[i]] == 0:
        del count[A[i]]
    A[i] = x
    count[x] += 1
    print(mex(count))