# Read the input
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Extract multiples of K and divide by K
multiples = [a // K for a in A if a % K == 0]

# Print the quotients in ascending order
print(" ".join(map(str, sorted(multiples))))