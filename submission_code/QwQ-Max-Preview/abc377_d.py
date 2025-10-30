n, m = map(int, input().split())
intervals = []
min_R = m  # Initialize with maximum possible R

for _ in range(n):
    l, r = map(int, input().split())
    intervals.append((l, r))
    if r < min_R:
        min_R = r

# Initialize max_L array to track maximum L for each R
max_L = [0] * (m + 2)  # indexes 0 to m+1

for l, r in intervals:
    if max_L[r] < l:
        max_L[r] = l

# Compute prefix maximum
current_max = 0
for r in range(1, m + 1):
    current_max = max(current_max, max_L[r])
    max_L[r] = current_max

sum_part1 = (min_R - 1) * min_R // 2
sum_part2 = 0

for r in range(min_R, m + 1):
    sum_part2 += max(0, r - max_L[r])

print(sum_part1 + sum_part2)