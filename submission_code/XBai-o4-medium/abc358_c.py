n, m = map(int, input().split())
stands = [input().strip() for _ in range(n)]

min_visits = n + 1  # Initialize with a value larger than possible

for mask in range(1, 1 << n):
    all_covered = True
    for flavor in range(m):
        found = False
        for stand in range(n):
            if (mask & (1 << stand)) and (stands[stand][flavor] == 'o'):
                found = True
                break
        if not found:
            all_covered = False
            break
    if all_covered:
        count = bin(mask).count('1')
        if count < min_visits:
            min_visits = count

print(min_visits)