import sys

# Read N: the number of people
n = int(sys.stdin.readline())

# Read the list P: the person numbers in the line from front to back.
# P_i is the person at the i-th position (1-based).
# The list `p` stores these numbers, 0-indexed. p[i] is the person at the (i+1)-th position.
p = list(map(int, sys.stdin.readline().split()))

# Read Q: the number of queries
q = int(sys.stdin.readline())

# Precompute the position (0-based index) of each person.
# We create a dictionary `pos` where keys are person numbers and values are their 0-based indices in the line.
# For example, if p = [2, 1, 3], then pos will be {2: 0, 1: 1, 3: 2}.
pos = {}
for index in range(n):
    person_number = p[index]
    pos[person_number] = index # Store the 0-based index

# List to store the results for each query
query_results = []

# Process each of the Q queries
for _ in range(q):
    # Read the two person numbers A_i and B_i for the current query
    a, b = map(int, sys.stdin.readline().split())

    # Retrieve the precomputed positions (indices) of person A and person B
    # from the `pos` dictionary. These lookups are O(1) on average.
    position_a = pos[a]
    position_b = pos[b]

    # Compare the positions. The person with the smaller index is standing further to the front.
    if position_a < position_b:
        # Person A has a smaller index, so person A is further to the front.
        query_results.append(str(a))
    else:
        # Person B has a smaller or equal index (equality won't happen due to distinct P_i).
        # So person B is further to the front.
        query_results.append(str(b))

# Print all the collected results. Each result should be on a new line.
# Using "
".join on the list of string results is an efficient way to format the output.
print("
".join(query_results))