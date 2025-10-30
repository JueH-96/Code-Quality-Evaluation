import bisect

t = int(input())
for _ in range(t):
    n, k = map(int, input().split())
    s = input().strip()
    blacks = [i for i, c in enumerate(s) if c == 'B']
    if not blacks:
        print(0)
        continue
    ops = 0
    current = 0
    while current < len(blacks):
        pos = blacks[current]
        end = min(pos + k - 1, n - 1)
        next_current = bisect.bisect_right(blacks, end)
        ops += 1
        current = next_current
    print(ops)