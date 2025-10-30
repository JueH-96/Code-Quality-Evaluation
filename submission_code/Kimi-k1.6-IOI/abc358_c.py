import itertools

n, m = map(int, input().split())
masks = []
for _ in range(n):
    s = input().strip()
    mask = 0
    for j in range(m):
        if s[j] == 'o':
            mask |= 1 << j
    masks.append(mask)

target = (1 << m) - 1

for k in range(1, n+1):
    for combo in itertools.combinations(masks, k):
        combined = 0
        for mask in combo:
            combined |= mask
            if combined == target:
                break
        if combined == target:
            print(k)
            break
    else:
        continue
    break