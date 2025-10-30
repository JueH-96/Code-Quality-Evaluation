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

full_mask = (1 << m) - 1

for k in range(1, n+1):
    for combo in itertools.combinations(range(n), k):
        combined = 0
        for idx in combo:
            combined |= masks[idx]
            if combined == full_mask:
                break  # Early exit if already full
        if combined == full_mask:
            print(k)
            break
    else:
        continue
    break