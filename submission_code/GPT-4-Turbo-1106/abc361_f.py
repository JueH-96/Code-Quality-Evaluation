import math

def count_integers(N):
    # Create a set to store unique powers
    unique_powers = set()
    
    # Upper limit for a is N^(1/2) since b >= 2
    max_a = int(N**0.5)
    
    # Iterate over all possible values of a
    for a in range(2, max_a + 1):
        # Start with the square of a and go up to N
        power = a * a
        while power <= N:
            unique_powers.add(power)
            power *= a
    
    # The result is the number of unique powers found
    return len(unique_powers)

# Read input from stdin
N = int(input().strip())

# Calculate and print the answer
print(count_integers(N))