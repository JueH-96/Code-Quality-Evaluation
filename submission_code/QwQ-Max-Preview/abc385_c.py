from collections import defaultdict

n = int(input())
h = list(map(int, input().split()))

groups = defaultdict(list)
for idx in range(n):
    height = h[idx]
    groups[height].append(idx + 1)  # 1-based position

max_result = 1  # Minimum possible is 1

for key in groups:
    positions = groups[key]
    m = len(positions)
    if m == 0:
        continue
    # Each element in dp represents the possible steps and their max lengths ending at that position
    dp = [defaultdict(int) for _ in range(m)]
    current_max = 1
    for i in range(m):
        for j in range(i):
            d = positions[i] - positions[j]
            # The length is 1 (for j's sequence) + 1 (current i)
            # If j doesn't have this d, default to 1 (j alone) +1 = 2
            current_length = dp[j].get(d, 1) + 1
            if current_length > dp[i][d]:
                dp[i][d] = current_length
                if current_length > current_max:
                    current_max = current_length
    if current_max > max_result:
        max_result = current_max

print(max_result)