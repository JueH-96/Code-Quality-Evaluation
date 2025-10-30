n = int(input())
a = list(map(int, input().split()))

# Create a mapping from value to sum of all elements greater than that value
from collections import defaultdict

# Count frequency of each value
freq = defaultdict(int)
for x in a:
    freq[x] += 1

# Sort unique values
unique_vals = sorted(freq.keys())

# For each value, compute sum of all elements greater than it
val_to_greater_sum = {}
suffix_sum = 0

# Process from largest to smallest
for val in reversed(unique_vals):
    val_to_greater_sum[val] = suffix_sum
    suffix_sum += val * freq[val]

# Output results
result = []
for ai in a:
    result.append(val_to_greater_sum[ai])

print(' '.join(map(str, result)))