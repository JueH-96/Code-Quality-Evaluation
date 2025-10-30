import math

def count_powers(N):
    unique_powers = set()
    
    # Iterate over possible exponents b starting from 2
    b = 2
    while True:
        # Calculate the maximum base a such that a^b <= N
        a_max = int(math.pow(N, 1.0 / b))
        if a_max < 2:
            break
        # Iterate over possible bases a from 2 to a_max
        for a in range(2, a_max + 1):
            x = a ** b
            if x <= N:
                unique_powers.add(x)
        b += 1
    
    # Include 1 since 1^b is 1 for any b >= 2
    unique_powers.add(1)
    
    return len(unique_powers)

# Read input
N = int(input())
# Compute and print the result
print(count_powers(N))