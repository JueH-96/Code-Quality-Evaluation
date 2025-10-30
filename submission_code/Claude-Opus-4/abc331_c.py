# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Create a sorted array with original indices
sorted_a = sorted([(a[i], i) for i in range(n)])

# Calculate prefix sums of the sorted array
prefix_sum = [0]
for val, _ in sorted_a:
    prefix_sum.append(prefix_sum[-1] + val)

# For each element, find sum of elements greater than it
result = [0] * n

for i in range(n):
    # Binary search to find the first element greater than a[i]
    left, right = 0, n
    while left < right:
        mid = (left + right) // 2
        if sorted_a[mid][0] <= a[i]:
            left = mid + 1
        else:
            right = mid
    
    # Sum of elements from position 'left' to end
    if left < n:
        result[i] = prefix_sum[n] - prefix_sum[left]

print(' '.join(map(str, result)))