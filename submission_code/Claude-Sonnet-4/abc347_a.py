# YOUR CODE HERE
# Read N and K
N, K = map(int, input().split())

# Read the sequence A
A = list(map(int, input().split()))

# Find multiples of K, divide by K, and collect quotients
quotients = []
for num in A:
    if num % K == 0:
        quotients.append(num // K)

# Print quotients with spaces in between
print(' '.join(map(str, quotients)))