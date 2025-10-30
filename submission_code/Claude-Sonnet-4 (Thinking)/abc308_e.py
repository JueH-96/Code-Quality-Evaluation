def mex(a, b, c):
    values = {a, b, c}
    for i in range(4):
        if i not in values:
            return i
    return 3

n = int(input())
a = list(map(int, input().split()))
s = input().strip()

# Precompute prefix counts of M's
prefix_m = [[0, 0, 0] for _ in range(n + 1)]
for i in range(n):
    prefix_m[i + 1] = prefix_m[i][:]
    if s[i] == 'M':
        prefix_m[i + 1][a[i]] += 1

# Precompute suffix counts of X's
suffix_x = [[0, 0, 0] for _ in range(n + 1)]
for i in range(n - 1, -1, -1):
    suffix_x[i] = suffix_x[i + 1][:]
    if s[i] == 'X':
        suffix_x[i][a[i]] += 1

total = 0

for j in range(n):
    if s[j] == 'E':
        m_counts = prefix_m[j]
        x_counts = suffix_x[j + 1]
        
        for val_i in range(3):
            for val_k in range(3):
                count = m_counts[val_i] * x_counts[val_k]
                if count > 0:
                    mex_value = mex(val_i, a[j], val_k)
                    total += count * mex_value

print(total)