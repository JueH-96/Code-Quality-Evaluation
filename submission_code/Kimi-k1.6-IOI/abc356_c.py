n, m, k = map(int, input().split())
tests = []
for _ in range(m):
    parts = input().split()
    c = int(parts[0])
    a_list = list(map(int, parts[1:-1]))
    r = parts[-1]
    mask = 0
    for a in a_list:
        mask |= 1 << (a - 1)
    tests.append((mask, r))

ans = 0
for s in range(1 << n):
    valid = True
    for (mask, r) in tests:
        cnt = bin(s & mask).count('1')
        if r == 'o' and cnt < k:
            valid = False
            break
        if r == 'x' and cnt >= k:
            valid = False
            break
    if valid:
        ans += 1
print(ans)