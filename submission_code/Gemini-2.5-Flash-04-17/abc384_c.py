# Read scores for problems A, B, C, D, E
import sys

# Read the five integer scores from standard input
a, b, c, d, e = map(int, sys.stdin.readline().split())

# Store scores and corresponding letters
scores = [a, b, c, d, e]
letters = ['A', 'B', 'C', 'D', 'E']

# List to store participant data as (score, name) tuples
participants = []

# The participants are defined by all non-empty subsequences of "ABCDE".
# This corresponds to all non-empty subsets of the problems {A, B, C, D, E}.
# There are 2^5 - 1 = 31 such subsets.
# We can represent each subset using a 5-bit mask from 1 to 31.
# Let's map bit positions to problems:
# Bit 4 (weight 16): Problem A (index 0 in scores/letters)
# Bit 3 (weight 8):  Problem B (index 1)
# Bit 2 (weight 4):  Problem C (index 2)
# Bit 1 (weight 2):  Problem D (index 3)
# Bit 0 (weight 1):  Problem E (index 4)

# Iterate through all possible non-empty masks (from 1 to 31)
for mask in range(1, 32):
    current_score = 0
    current_name = ""
    
    # Check each problem from A to E (index i from 0 to 4)
    for i in range(5):
        # The bit position corresponding to problem letters[i] is 4-i.
        # E.g., for i=0 (Problem A), bit position is 4. For i=4 (Problem E), bit position is 0.
        bit_position = 4 - i
        
        # If the corresponding bit is set in the mask, the participant solved this problem
        if (mask >> bit_position) & 1:
            # Add the score for this problem
            current_score += scores[i]
            # Append the letter to the participant's name.
            # Since we iterate i from 0 to 4, letters are added in alphabetical order (A, B, C, D, E).
            current_name += letters[i]

    # Store the participant's data. We use a tuple (-score, name)
    # The negative score ensures that sorting in ascending order of the first element
    # results in sorting by score in descending order.
    # If scores are equal, the second element (name) is used for sorting,
    # which handles the lexicographical tie-breaking rule correctly.
    participants.append((-current_score, current_name))

# Sort the participants list based on the tuples (-score, name)
# This sorts by score descending, then by name lexicographically ascending.
participants.sort()

# Print the names of the participants in the sorted order
# We iterate through the sorted list and print the second element of each tuple (the name).
for _, name in participants:
    print(name)