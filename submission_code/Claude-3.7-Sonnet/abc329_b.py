# Read the number of integers
n = int(input().strip())

# Read the list of integers
a = list(map(int, input().strip().split()))

# Find the maximum value
max_value = max(a)

# Filter out all occurrences of the maximum value
filtered_a = [x for x in a if x != max_value]

# Find the maximum of the filtered list (the largest non-maximum value)
second_largest = max(filtered_a)

# Print the result
print(second_largest)