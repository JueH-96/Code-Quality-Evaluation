# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

total_sum = sum(a)
target_low = total_sum // n
target_high = target_low + 1
num_high = total_sum % n  # number of elements that should be target_high
num_low = n - num_high    # number of elements that should be target_low

# Sort the array to optimally assign targets
a.sort()

operations = 0

# First num_low elements should be target_low
for i in range(num_low):
    if a[i] > target_low:
        operations += a[i] - target_low

# Remaining elements should be target_high  
for i in range(num_low, n):
    if a[i] > target_high:
        operations += a[i] - target_high

print(operations)