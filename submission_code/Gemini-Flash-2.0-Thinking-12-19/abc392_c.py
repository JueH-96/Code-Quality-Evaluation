# YOUR CODE HERE
import sys

# Read N
n = int(sys.stdin.readline())

# Read P values
# P_i is the person that person i stares at. Stored in p_list[i-1].
# The input gives 1-indexed person IDs, which are stored as values in the list.
p_list = list(map(int, sys.stdin.readline().split()))

# Read Q values
# Q_i is the bib number of person i. Stored in q_list[i-1].
# The input gives 1-indexed bib numbers, which are stored as values in the list.
q_list = list(map(int, sys.stdin.readline().split()))

# Create a mapping from bib number (1-indexed) to the person ID (1-indexed) wearing that bib.
# person_from_bib[bib_num] = person_id
# This is the inverse mapping of Q. If person i (1-indexed) wears bib Q_i (1-indexed),
# then bib Q_i is worn by person i.
# We use a list of size N+1 for 1-based indexing for bib numbers.
person_from_bib = [0] * (n + 1)
for i in range(n):
    # Iterate through person indices 0 to N-1, representing persons 1 to N.
    # Person i+1 (1-indexed) wears bib number q_list[i] (1-indexed).
    # So, the person wearing bib q_list[i] is i+1.
    person_from_bib[q_list[i]] = i + 1

# Calculate S_k for each bib number k from 1 to N.
# S_k is the bib number of the person that the person wearing bib number k is staring at.
# We store the results in s_list, where s_list[k-1] will hold the result for bib number k.
s_list = [0] * n

# Iterate through each bib number k from 1 to N.
for k in range(1, n + 1):
    # Step 1: Find the person wearing bib number k.
    # We use the person_from_bib map to get the person ID (1-indexed) from the bib number k.
    # Let this person ID be `current_person_id`.
    current_person_id = person_from_bib[k]

    # Step 2: Find the person that `current_person_id` is staring at.
    # The input P tells us that person i (1-indexed) stares at person P_i (1-indexed).
    # In our 0-indexed p_list, person i+1 (1-indexed) stares at person p_list[i] (1-indexed).
    # So, `current_person_id` (1-indexed) stares at the person whose ID is p_list[current_person_id - 1] (1-indexed).
    # Let this person ID be `stared_person_id`.
    stared_person_id = p_list[current_person_id - 1]

    # Step 3: Find the bib number of `stared_person_id`.
    # The input Q tells us that person i (1-indexed) wears bib number Q_i (1-indexed).
    # In our 0-indexed q_list, person i+1 (1-indexed) wears bib number q_list[i] (1-indexed).
    # So, `stared_person_id` (1-indexed) wears the bib number q_list[stared_person_id - 1] (1-indexed).
    # Let this be `bib_of_stared_person`.
    bib_of_stared_person = q_list[stared_person_id - 1]

    # Step 4: Store the result S_k.
    # The result `bib_of_stared_person` corresponds to the bib number k.
    # We store this result in `s_list` at the index corresponding to bib k.
    # Since `s_list` is 0-indexed and we're calculating for bib number k (1-indexed),
    # we store it at index k-1.
    s_list[k - 1] = bib_of_stared_person

# Print the elements of s_list separated by spaces.
# The *s_list syntax unpacks the list elements as arguments to print.
print(*s_list)