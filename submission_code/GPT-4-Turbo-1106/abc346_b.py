# Read input
W, B = map(int, input().split())

# Define the pattern of the piano keys
pattern = "wbwbwwbwbwbw"

# Check if there is a substring with W white keys and B black keys
def check_substring(W, B, pattern):
    # Calculate the total length of the substring we are looking for
    total_length = W + B
    
    # Iterate over the pattern to find a matching substring
    for start in range(len(pattern)):
        # Counters for white and black keys
        white_count = 0
        black_count = 0
        
        # Check the substring starting at 'start' and of length 'total_length'
        for i in range(total_length):
            # Calculate the index in the pattern, considering it repeats infinitely
            index = (start + i) % len(pattern)
            
            # Count white and black keys
            if pattern[index] == 'w':
                white_count += 1
            else:
                black_count += 1
            
            # If we have the required number of white and black keys, return True
            if white_count == W and black_count == B:
                return True
            
            # If we have more than the required number of either key, break early
            if white_count > W or black_count > B:
                break
    
    # If no matching substring is found, return False
    return False

# Print the result
print("Yes" if check_substring(W, B, pattern) else "No")