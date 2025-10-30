# Read the input values
n, k = map(int, input().split())
a = list(map(int, input().split()))

# Create the new list by taking the last k elements and then the rest
result = a[-k:] + a[:-k]

# Print the result separated by spaces
print(' '.join(map(str, result)))