# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Precompute prefix distinct counts
prefix_distinct = [0] * (n + 1)
seen = set()
for i in range(n):
    seen.add(a[i])
    prefix_distinct[i + 1] = len(seen)

# Precompute suffix distinct counts
suffix_distinct = [0] * (n + 1)
seen = set()
for i in range(n - 1, -1, -1):
    seen.add(a[i])
    suffix_distinct[i] = len(seen)

max_sum = 0

# Try all possible (i, j) pairs
for i in range(1, n - 1):
    # For each i, calculate distinct count in middle subarray for all valid j
    seen = set()
    for j in range(i, n - 1):
        seen.add(a[j])
        
        # Count distinct in three parts
        left_count = prefix_distinct[i]
        middle_count = len(seen)
        right_count = suffix_distinct[j + 1]
        
        total = left_count + middle_count + right_count
        max_sum = max(max_sum, total)

print(max_sum)