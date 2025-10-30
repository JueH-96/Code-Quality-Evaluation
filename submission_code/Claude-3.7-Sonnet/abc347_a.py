# Read input
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Filter multiples of K, divide by K, and sort
quotients = [a // K for a in A if a % K == 0]
quotients.sort()

# Print the result
print(' '.join(map(str, quotients)))