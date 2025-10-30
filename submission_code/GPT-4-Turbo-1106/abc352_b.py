# Read the input strings S and T
S = input().strip()
T = input().strip()

# Initialize the index for S and an empty list for the correct positions
index_s = 0
correct_positions = []

# Iterate through the characters in T
for index_t, char_t in enumerate(T):
    # If the character in T matches the current character in S
    if index_s < len(S) and char_t == S[index_s]:
        # Add the position (index + 1) to the list of correct positions
        correct_positions.append(index_t + 1)
        # Move to the next character in S
        index_s += 1

# Print the correct positions separated by spaces
print(' '.join(map(str, correct_positions)))