# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Compute distinct count for prefixes
# prefix_distinct[i] = count of distinct elements in a[0:i]
prefix_distinct = [0] * (n + 1)
seen = set()
for i in range(n):
    seen.add(a[i])
    prefix_distinct[i + 1] = len(seen)

# Compute distinct count for suffixes
# suffix_distinct[i] = count of distinct elements in a[i:n]
suffix_distinct = [0] * (n + 1)
seen = set()
for i in range(n - 1, -1, -1):
    seen.add(a[i])
    suffix_distinct[i] = len(seen)

# Find the maximum sum
max_sum = 0
for i in range(1, n):
    # Split at position i (0-indexed)
    # Left subarray: a[0:i], Right subarray: a[i:n]
    left_count = prefix_distinct[i]
    right_count = suffix_distinct[i]
    max_sum = max(max_sum, left_count + right_count)

print(max_sum)