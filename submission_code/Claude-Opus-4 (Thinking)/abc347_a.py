# YOUR CODE HERE
# Read N and K
n, k = map(int, input().split())

# Read the array A
a = list(map(int, input().split()))

# Extract multiples of K and divide by K
quotients = []
for num in a:
    if num % k == 0:
        quotients.append(num // k)

# Print the quotients
print(' '.join(map(str, quotients)))