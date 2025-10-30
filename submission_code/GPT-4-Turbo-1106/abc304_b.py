# Read the integer from standard input
N = int(input())

# Define the truncation levels
truncation_levels = [
    (10**3, 1),
    (10**4, 10),
    (10**5, 100),
    (10**6, 1000),
    (10**7, 10000),
    (10**8, 100000),
    (10**9, 1000000)
]

# Determine the truncation level for N
for level, truncation in truncation_levels:
    if N < level:
        N = N // truncation * truncation
        break

# Print the result
print(N)