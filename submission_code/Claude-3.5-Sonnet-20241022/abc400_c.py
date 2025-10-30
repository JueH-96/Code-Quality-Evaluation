def count_good_integers(N):
    # For each number X from 1 to N, X = 2^a * b^2
    # where a ≥ 0 and b ≥ 1
    result = 0
    
    # Try each power of 2 (a)
    power_of_2 = 1
    while power_of_2 <= N:
        # For this power of 2, find how many perfect squares we can multiply with
        # to get a number ≤ N
        
        # If X = 2^a * b^2 ≤ N
        # then b^2 ≤ N/2^a
        max_square = N // power_of_2
        
        # Count how many perfect squares are there up to max_square
        # b^2 ≤ max_square
        # b ≤ sqrt(max_square)
        count = int((max_square ** 0.5))
        
        result += count
        
        # Move to next power of 2
        power_of_2 *= 2
        
        # If power_of_2 overflows, break
        if power_of_2 > N:
            break
    
    return result

# Read input
N = int(input())

# Print output
print(count_good_integers(N))