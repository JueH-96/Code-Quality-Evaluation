# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Precompute distinct count from left
left_distinct = [0] * n
seen_left = set()
for i in range(n):
    seen_left.add(a[i])
    left_distinct[i] = len(seen_left)

# Precompute distinct count from right
right_distinct = [0] * n
seen_right = set()
for i in range(n-1, -1, -1):
    seen_right.add(a[i])
    right_distinct[i] = len(seen_right)

# Find maximum sum
max_sum = 0
for i in range(n-1):
    # Split after position i (0-indexed)
    # Left part: indices 0 to i
    # Right part: indices i+1 to n-1
    current_sum = left_distinct[i] + right_distinct[i+1]
    max_sum = max(max_sum, current_sum)

print(max_sum)