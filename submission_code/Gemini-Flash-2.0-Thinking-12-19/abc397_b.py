# YOUR CODE HERE
import sys

def solve():
    """
    Calculates the minimum number of characters to insert into string S
    to make it conform to the pattern "ioioio..." with even length.
    """
    # Read the input string from standard input.
    S = sys.stdin.readline().strip()
    
    # current_pattern_pos tracks the current position in the target pattern "ioioio..."
    # that we are trying to fill. It also implicitly represents the current length
    # of the string being built by combining characters from S and insertions.
    # We use 0-based indexing for positions: 0, 1, 2, 3, ...
    current_pattern_pos = 0
    
    # s_index tracks the index of the current character in the input string S
    # that we are considering placing in the constructed string.
    s_index = 0
    
    # insertions counts the minimum characters we have inserted so far.
    insertions = 0
    
    # We iterate through the input string S.
    # The loop continues as long as there are characters left in S to process.
    while s_index < len(S):
        # Determine the character required at the current position in the target pattern.
        # In 0-indexed positions:
        # Position 0 needs 'i', Position 1 needs 'o',
        # Position 2 needs 'i', Position 3 needs 'o', and so on.
        # Generally, position k needs 'i' if k is even, and 'o' if k is odd.
        required_char = 'i' if current_pattern_pos % 2 == 0 else 'o'
        
        # Check if the current character from S matches the required character at
        # the current pattern position we are trying to fill.
        if S[s_index] == required_char:
            # The character S[s_index] matches the requirement for the current
            # pattern position (current_pattern_pos). We can use this character
            # from S at this position.
            
            # Move to consider the next character in S.
            s_index += 1
            # Move to the next position in the target pattern, which is now conceptually filled.
            current_pattern_pos += 1
        else:
            # The character S[s_index] does NOT match the required character at
            # the current pattern position (current_pattern_pos).
            # To maintain the "ioio..." pattern structure while preserving the
            # relative order of characters in S, we must conceptually insert
            # the 'required_char' at this position.
            
            # Increment the insertion count.
            insertions += 1
            # The current pattern position (current_pattern_pos) is now conceptually
            # filled by the insertion. Move to the next position in the pattern.
            current_pattern_pos += 1
            # The character S[s_index] was not used for this position; it is still
            # the next character from S to consider. So, s_index does NOT increment.
            # It will be evaluated against the requirement at the new current_pattern_pos
            # in the next iteration.

    # After the loop finishes (meaning s_index has reached len(S), so all
    # characters from S have been considered and conceptually placed),
    # current_pattern_pos represents the total number of positions covered in the
    # target pattern. This is the length of the string formed by S plus insertions.
    
    # The problem requires the final string to have an even length.
    # If the current total length constructed so far (current_pattern_pos) is odd,
    # we need to add one more character to make the total length even.
    # This additional character must follow the "ioio..." pattern.
    # The current length is current_pattern_pos. If it's odd, the last character
    # added/matched was at 0-indexed position `current_pattern_pos - 1` (which is even)
    # and was 'i'. The sequence ends "...i". To get an even length string
    # following the pattern, we must add an 'o' at position `current_pattern_pos`
    # (which is odd 0-indexed, corresponding to an even 1-indexed position).
    # This requires one more insertion.
    # If the current length (current_pattern_pos) is already even, the last
    # character added/matched was at 0-indexed position `current_pattern_pos - 1`
    # (which is odd) and was 'o'. The sequence ends "...o". The length is even,
    # and the last character is 'o' (at an odd 0-indexed position / even 1-indexed position).
    # This satisfies the even length requirement and the pattern ending.
    # No additional insertion is needed in this case.
    if current_pattern_pos % 2 != 0:
        insertions += 1

    # Print the final minimum number of characters that needed to be inserted.
    print(insertions)

# Execute the solve function to read input, perform calculation, and print output.
solve()