# Read input
N = int(input())
garbage = []
for _ in range(N):
    q, r = map(int, input().split())
    garbage.append((q, r))
Q = int(input())
for _ in range(Q):
    t, d = map(int, input().split())
    q, r = garbage[t-1]
    rem = d % q
    if rem == r:
        print(d)
    else:
        if rem < r:
            next_day = d + (r - rem)
        else:
            next_day = d + (q - (rem - r))
        print(next_day)