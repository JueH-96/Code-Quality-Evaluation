import sys

# Read N
N = int(sys.stdin.readline())

# Read scores P_1 P_2 ... P_N
# The input P_i corresponds to the score of person i (1-based index).
# We store them in a list P where P[i] is the score of person i+1 (0-based index i).
# Using 0-based indexing internally for lists is standard Python practice.
P = list(map(int, sys.stdin.readline().split()))

# Create a list of (score, original_index) tuples.
# original_index is the 0-based index (0 to N-1) corresponding to the input order.
# This allows us to recover the original person's index after sorting.
participants = []
for i in range(N):
    participants.append((P[i], i)) # Store (score, original_0_based_index)

# Sort participants by score in descending order.
# This is the key step to process people from highest score to lowest.
# If scores are equal, the relative order doesn't matter for the rank calculation
# procedure described, as they all get the same rank. Python's sort is stable,
# which is fine.
sorted_participants = sorted(participants, key=lambda item: item[0], reverse=True)

# Calculate ranks.
# Initialize a list 'ranks' where ranks[i] will store the rank of the person
# whose score was originally P[i]. This list is indexed by the original 0-based index.
ranks = [0] * N
r = 1 # Initialize the current rank to assign (starts at 1)
i = 0 # Index in the sorted_participants list, representing the start of the current score group

# Iterate through the sorted participants group by group (by equal scores)
# This loop processes all participants exactly once.
while i < N:
    # Get the score of the current group leader (the first participant in the sorted block)
    current_score = sorted_participants[i][0]

    # Find the end index 'j' of the block of participants with the same score
    # starting from index 'i'.
    j = i
    while j < N and sorted_participants[j][0] == current_score:
        j += 1

    # The number of participants in this block (who have the same current_score)
    k = j - i

    # Assign the current rank 'r' to all participants in this block.
    # These participants are from index i up to j-1 in the sorted_participants list.
    for l in range(i, j):
        original_index = sorted_participants[l][1] # Get the original 0-based index
        ranks[original_index] = r # Assign the rank to the correct position in the ranks list

    # Update the current rank 'r' for the next block of participants.
    # The next block starts receiving rank r + k.
    r += k

    # Move the starting index 'i' to the beginning of the next block.
    # The next iteration will process the group starting at index j.
    i = j

# Print the calculated ranks for each person in the original order (person 1 to N).
# The list 'ranks' is already ordered according to the original input:
# ranks[0] is the rank of person 1 (score P[0]), ranks[1] is rank of person 2 (score P[1]), etc.
for i in range(N):
    print(ranks[i])