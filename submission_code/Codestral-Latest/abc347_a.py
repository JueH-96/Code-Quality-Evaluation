# YOUR CODE HERE
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Extract elements that are multiples of K and divide them by K
quotients = [x // K for x in A if x % K == 0]

# Print the quotients in ascending order
print(' '.join(map(str, quotients)))