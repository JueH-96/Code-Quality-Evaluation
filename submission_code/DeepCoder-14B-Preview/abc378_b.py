n = int(input())
q = []
r = []
for _ in range(n):
    a, b = map(int, input().split())
    q.append(a)
    r.append(b)

Q = int(input())
for _ in range(Q):
    t, d = map(int, input().split())
    t -= 1  # convert to 0-based index
    qi = q[t]
    ri = r[t]
    current_mod = d % qi
    delta = (ri - current_mod) % qi
    next_day = d + delta
    print(next_day)