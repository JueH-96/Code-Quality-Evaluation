# YOUR CODE HERE
def min_insertions(S):
    insertions = 0
    pattern_pos = 0
    
    for c in S:
        # Determine the expected character at the current position
        expected_char = 'i' if pattern_pos % 2 == 0 else 'o'
        
        # If the character doesn't match the expected, insert a character
        if c != expected_char:
            insertions += 1
            pattern_pos += 1
            # Recalculate expected character after incrementing position
            expected_char = 'i' if pattern_pos % 2 == 0 else 'o'
        
        # Move to the next position
        pattern_pos += 1
    
    # If final position is odd, we need to add one more character to make it even
    if pattern_pos % 2 == 1:
        insertions += 1
    
    return insertions

# Read input from stdin
S = input().strip()

# Print the answer
print(min_insertions(S))