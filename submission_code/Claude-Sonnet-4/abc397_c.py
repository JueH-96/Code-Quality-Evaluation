# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Precompute distinct counts for prefixes (left to right)
left_distinct = [0] * n
seen_left = set()
for i in range(n):
    seen_left.add(a[i])
    left_distinct[i] = len(seen_left)

# Precompute distinct counts for suffixes (right to left)
right_distinct = [0] * n
seen_right = set()
for i in range(n-1, -1, -1):
    seen_right.add(a[i])
    right_distinct[i] = len(seen_right)

# Find maximum sum for all possible splits
max_sum = 0
for i in range(1, n):
    # Split at position i means:
    # Left part: a[0] to a[i-1] (inclusive)
    # Right part: a[i] to a[n-1] (inclusive)
    left_count = left_distinct[i-1]
    right_count = right_distinct[i]
    max_sum = max(max_sum, left_count + right_count)

print(max_sum)