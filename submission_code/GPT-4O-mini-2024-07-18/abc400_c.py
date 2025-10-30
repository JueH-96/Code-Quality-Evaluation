def count_good_integers(N):
    count = 0
    
    # Iterate over all possible values of b
    b = 1
    while b * b <= N:
        # Calculate b^2
        b_squared = b * b
        
        # Now we need to find the maximum a such that 2^a * b^2 <= N
        # This means we need to find the largest power of 2 that fits
        # We can find this by checking how many times we can multiply b_squared by 2
        power_of_2 = 1
        while power_of_2 * b_squared <= N:
            count += 1
            power_of_2 *= 2
        
        b += 1
    
    return count

import sys

# Read input
N = int(sys.stdin.read().strip())

# Get the count of good integers
result = count_good_integers(N)

# Print the result
print(result)