n, m, k = map(int, input().split())
tests = []

for _ in range(m):
    parts = input().split()
    c_i = int(parts[0])
    keys = list(map(int, parts[1:1 + c_i]))
    r_i = parts[1 + c_i]
    mask = 0
    for key in keys:
        mask |= (1 << (key - 1))
    tests.append((mask, r_i))

total = 0
for subset in range(1 << n):
    valid = True
    for (mask, r) in tests:
        common = subset & mask
        count = bin(common).count('1')
        if r == 'o':
            if count < k:
                valid = False
                break
        else:
            if count >= k:
                valid = False
                break
    if valid:
        total += 1

print(total)