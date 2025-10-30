import sys

# YOUR CODE HERE
def solve():
    S = sys.stdin.readline().strip()

    # Step 1: Create a mapping from character to its coordinate
    # The coordinate for S[i] is (i + 1) because coordinates are 1-indexed
    char_to_coord = {}
    for i in range(len(S)):
        char_to_coord[S[i]] = i + 1

    # Step 2: Define the sequence of characters to type
    typing_sequence = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Step 3: Initialize total distance and current finger position
    total_distance = 0
    # The finger starts at the coordinate of 'A'
    current_pos = char_to_coord['A']

    # Step 4: Iterate through the typing sequence from the second character ('B') to the last ('Z')
    # and calculate the distance moved for each step.
    # We iterate from index 1 to the end of the typing_sequence string.
    for i in range(1, len(typing_sequence)):
        next_char = typing_sequence[i]
        next_pos = char_to_coord[next_char]
        
        # Add the absolute distance between the current position and the next character's position
        total_distance += abs(next_pos - current_pos)
        
        # Update the current position to the position of the character that was just "typed"
        current_pos = next_pos

    # Step 5: Print the total calculated distance
    print(total_distance)

# Call the solve function to run the program
solve()