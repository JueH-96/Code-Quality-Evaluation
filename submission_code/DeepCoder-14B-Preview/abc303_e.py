n = int(input())
degrees = [0] * (n + 1)
for _ in range(n - 1):
    u, v = map(int, input().split())
    degrees[u] += 1
    degrees[v] += 1

# Collect degrees of nodes with degree >= 2
deg_list = [d for d in degrees[1:] if d >= 2]
deg_list.sort(reverse=True)

# Find the correct K
levels = []
for k in range(1, len(deg_list) + 1):
    if k > len(deg_list):
        continue
    current_sum = sum(deg_list[:k])
    if current_sum == n - k:
        levels = deg_list[:k]
        break

# Sort the levels in ascending order
levels.sort()
print(' '.join(map(str, levels)))