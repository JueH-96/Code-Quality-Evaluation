# Read input
A, B = map(int, input().split())

# Calculate N
N = A // B
if A % B != 0:
    N += 1

# Print the result
print(N)