# Read input values
n, k = map(int, input().split())
a = list(map(int, input().split()))

# Process the list to find elements divisible by K and divide them
result = [str(x // k) for x in a if x % k == 0]

# Print the result
print(' '.join(result))