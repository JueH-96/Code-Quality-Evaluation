n = int(input())
garbage = []
for _ in range(n):
    q, r = map(int, input().split())
    garbage.append((q, r))
q = int(input())
for _ in range(q):
    t, d = map(int, input().split())
    t -= 1  # convert to 0-based index
    qi, ri = garbage[t]
    rem = d % qi
    base = d - rem
    if rem <= ri:
        x = base + ri
    else:
        x = base + qi + ri
    print(x)