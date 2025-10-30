import math

def count_powers(N):
    unique_powers = set()
    # Iterate over possible exponents b starting from 2
    b = 2
    while True:
        # Calculate the maximum base a such that a^b <= N
        max_a = int(math.pow(N, 1 / b))
        if max_a < 1:
            break
        # Iterate over possible bases a
        for a in range(2, max_a + 1):
            x = a ** b
            if x > N:
                break
            unique_powers.add(x)
        b += 1
    # Include 1 since 1^b is always 1 for any b >= 2
    unique_powers.add(1)
    return len(unique_powers)

# Read input
N = int(input())
# Compute and print the result
print(count_powers(N))