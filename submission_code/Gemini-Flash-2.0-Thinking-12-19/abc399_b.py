import sys

# Read N
N = int(sys.stdin.readline())

# Read scores P_1, P_2, ..., P_N
# Store them in a list P, where P[i] corresponds to the score of the person
# who was the (i+1)-th entry in the input. We use 0-based indexing for lists.
P = list(map(int, sys.stdin.readline().split()))

# Create a list of tuples (score, original_index)
# The original_index is 0-based (ranging from 0 to N-1), which corresponds
# to the person's position in the original input list.
people_with_indices = []
for i in range(N):
    people_with_indices.append((P[i], i))

# Sort the list by score in descending order.
# If scores are tied, their relative order after sorting doesn't affect the rank calculation,
# as all tied scores receive the same rank.
people_with_indices.sort(key=lambda item: item[0], reverse=True)

# Initialize the ranks list. ranks[i] will store the calculated rank of the person
# who was originally at index i in the input (corresponding to person i+1).
ranks = [0] * N

# Calculate ranks according to the procedure described in the problem.
# 'current_rank_to_assign' keeps track of the rank that will be assigned to the
# next group of people with the same score. It starts at 1.
current_rank_to_assign = 1
i = 0 # Pointer to the current position in the sorted list people_with_indices

# Iterate through the sorted list, processing people in groups of equal scores.
while i < N:
    # Get the score of the person at the current position in the sorted list.
    current_score = people_with_indices[i][0]

    # Find the end of the current group: all people with the same current_score.
    # 'j' will be the index of the first person *after* the current group in the sorted list.
    j = i
    while j < N and people_with_indices[j][0] == current_score:
        j += 1

    # The number of people in the current group with the same score is the difference between j and i.
    num_in_group = j - i

    # Assign the 'current_rank_to_assign' to all people in this group.
    # Iterate through the people from index 'i' up to 'j-1' in the sorted list.
    for l in range(i, j):
        # Get the original 0-based index of this person from the tuple.
        original_index = people_with_indices[l][1]
        # Assign the current rank to this person in the results array using their original index.
        ranks[original_index] = current_rank_to_assign

    # Update the rank for the next group. The rank increases by the number of people
    # who were just assigned a rank (the size of the current group).
    current_rank_to_assign += num_in_group

    # Move the pointer 'i' to the beginning of the next group.
    i = j

# Print the calculated ranks for each person in their original input order (person 1, person 2, ...).
# Since ranks[i] stores the rank of the person originally at index i (person i+1),
# printing the ranks array sequentially gives the required output.
for rank in ranks:
    print(rank)