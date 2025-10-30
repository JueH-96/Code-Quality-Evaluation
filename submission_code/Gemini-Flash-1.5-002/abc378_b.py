N = int(input())
qr = []
for _ in range(N):
    q, r = map(int, input().split())
    qr.append((q, r))

Q = int(input())
for _ in range(Q):
    t, d = map(int, input().split())
    q, r = qr[t - 1]
    if d % q == r:
        print(d)
    else:
        next_day = d + (r - d % q) % q
        print(next_day)