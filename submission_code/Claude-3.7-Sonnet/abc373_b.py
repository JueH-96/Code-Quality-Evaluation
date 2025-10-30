def min_distance(S):
    # Map from character to its position in S
    char_to_pos = {char: i for i, char in enumerate(S, 1)}
    
    # Calculate total distance
    total_distance = 0
    prev_pos = char_to_pos['A']
    
    # Process each character from B to Z
    for i in range(1, 26):  # From B to Z
        curr_char = chr(ord('A') + i)
        curr_pos = char_to_pos[curr_char]
        total_distance += abs(curr_pos - prev_pos)
        prev_pos = curr_pos
    
    return total_distance

# Read input
S = input().strip()

# Calculate and print the result
print(min_distance(S))