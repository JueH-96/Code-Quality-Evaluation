# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Find the maximum value
max_val = max(a)

# Filter out all values equal to the maximum
not_max = [x for x in a if x != max_val]

# Find the maximum of the remaining values
result = max(not_max)

print(result)