# Read input
N = int(input())
garbage_types = []
for _ in range(N):
    q, r = map(int, input().split())
    garbage_types.append((q, r))

Q = int(input())
for _ in range(Q):
    t, d = map(int, input().split())
    q, r = garbage_types[t-1]
    rem = d % q
    delta = (r - rem) % q
    print(d + delta)