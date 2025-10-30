# Read input from stdin
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Extract multiples of K and divide them by K
multiples = [a // K for a in A if a % K == 0]

# Print the quotients in ascending order
print(*sorted(multiples))