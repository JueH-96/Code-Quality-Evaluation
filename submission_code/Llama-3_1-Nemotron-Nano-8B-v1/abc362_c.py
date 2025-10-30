import sys

n = int(sys.stdin.readline())
L = []
R = []
sum_L = 0
sum_R = 0

for _ in range(n):
    a, b = map(int, sys.stdin.readline().split())
    L.append(a)
    R.append(b)
    sum_L += a
    sum_R += b

if sum_L > 0 or sum_R < 0:
    print("No")
else:
    delta = -sum_L
    X = L.copy()
    for i in range(n):
        if delta <= 0:
            break
        add = min(delta, R[i] - X[i])
        X[i] += add
        delta -= add
    print("Yes")
    print(' '.join(map(str, X)))