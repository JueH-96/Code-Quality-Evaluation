from bisect import bisect_right

N, M = map(int, input().split())
A = sorted(map(int, input().split()))

if M >= sum(A):
    print('infinite')
    exit()

acc = [0]
for a in A:
    acc.append(acc[-1] + a)

def ok(x):
    if x == 0:
        return True
    i = bisect_right(A, x)
    return (N - i) * x + acc[i] <= M

l, r = 0, 10**9
while r - l > 1:
    m = (l + r) // 2
    if ok(m):
        l = m
    else:
        r = m

print(l)