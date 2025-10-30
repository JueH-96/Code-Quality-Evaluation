import bisect

# Read input
n = int(input())
a = list(map(int, input().split()))

# Create a sorted copy of the array
sorted_a = sorted(a)

# Compute cumulative sums
cum_sum = [0] * (n + 1)
for i in range(n):
    cum_sum[i + 1] = cum_sum[i] + sorted_a[i]

# Calculate the sum of elements greater than each element
result = []
for val in a:
    # Find the position of the first element greater than val
    pos = bisect.bisect_right(sorted_a, val)
    
    # Sum of elements greater than val = total sum - sum of elements <= val
    sum_greater = cum_sum[n] - cum_sum[pos]
    result.append(sum_greater)

# Output the result
print(" ".join(map(str, result)))