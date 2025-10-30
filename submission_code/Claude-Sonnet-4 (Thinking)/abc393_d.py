n = int(input())
s = input().strip()

# Find positions of all 1s (0-indexed)
ones_positions = []
for i in range(n):
    if s[i] == '1':
        ones_positions.append(i)

k = len(ones_positions)

# Compute qi = pi - i
q = []
for i in range(k):
    q.append(ones_positions[i] - i)

# Find median of q
q.sort()
median = q[(k - 1) // 2]

L = median

# Calculate cost
cost = 0
for i in range(k):
    target_pos = L + i
    cost += abs(ones_positions[i] - target_pos)

print(cost)