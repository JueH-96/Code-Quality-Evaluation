# Read N and K
N, K = map(int, input().split())

# Read the sequence A
A = list(map(int, input().split()))

# Find multiples of K and compute quotients
quotients = []
for a in A:
    if a % K == 0:
        quotients.append(a // K)

# Print quotients with spaces in between
print(' '.join(map(str, quotients)))