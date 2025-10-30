# YOUR CODE HERE
import sys

# Read input
# N: number of programmers (2 <= N <= 50)
# M: number of superiority relations (0 <= M <= N*(N-1)/2)
N, M = map(int, sys.stdin.readline().split())

# Initialize adjacency matrix (0-based indexing)
# is_stronger[i][j] is True if person i+1 is stronger than person j+1
# Initialize all entries to False
is_stronger = [[False for _ in range(N)] for _ in range(N)]

# Read M superiority relations and populate the matrix
# A_i is stronger than B_i means there is a directed edge from A_i to B_i
for _ in range(M):
    u, v = map(int, sys.stdin.readline().split())
    # Adjust to 0-based indexing for the matrix
    is_stronger[u - 1][v - 1] = True

# Compute transitive closure using Floyd-Warshall algorithm
# For every intermediate node k (from 0 to N-1):
# Consider all pairs (i, j).
# If there is a path from i to k AND a path from k to j, then there is a path from i to j.
# This translates to: If person i is stronger than k AND person k is stronger than j,
# then person i is also stronger than j.
for k in range(N):
    for i in range(N):
        for j in range(N):
            # If person i is stronger than k AND person k is stronger than j
            if is_stronger[i][k] and is_stronger[k][j]:
                # Then person i is also stronger than j. Set is_stronger[i][j] to True.
                # This performs the core step of transitive closure.
                is_stronger[i][j] = True

# Find potential strongest programmers
# A person p is the strongest if they are stronger than ALL other N-1 people.
# We look for a person p for whom is_stronger[p][q] is True for every q != p
# based on the transitive closure.
potential_strongest = []
for p in range(N): # Iterate through each person p (0-indexed)
    # Assume person p (index p) is potentially the unique strongest
    is_potential_unique_strongest = True
    
    # Check if person p is stronger than every other person q
    for q in range(N): # Iterate through every other person q (0-indexed)
        # Skip comparing a person to themselves; the relation is between distinct programmers.
        if p == q:
            continue
        
        # If person p is NOT proven stronger than person q based on the transitive closure,
        # then p cannot be the unique strongest programmer.
        # The condition for p to be the unique strongest is that p must be stronger
        # than ALL other q's based on the available information and transitivity.
        # We check if the relation p > q is *not* established by the transitive closure.
        if not is_stronger[p][q]:
            is_potential_unique_strongest = False
            break # No need to check other q's for this person p; p is not the unique strongest candidate.
            
    # If after checking all other people q, person p is proven stronger than all of them
    # (i.e., is_potential_unique_strongest remains True), then this person is a candidate
    # for the unique strongest programmer because their superiority over everyone else
    # is necessitated by the given information.
    if is_potential_unique_strongest:
        # Add the 1-based index of this person to the list of potential candidates.
        potential_strongest.append(p + 1)

# Check the number of potential unique strongest programmers found
if len(potential_strongest) == 1:
    # If exactly one person is found who is proven stronger than everyone else,
    # that person is the uniquely determined strongest programmer across all
    # consistent total orderings.
    print(potential_strongest[0])
else:
    # If zero persons are proven stronger than everyone else, or if more than one
    # person is proven stronger than everyone else (the latter case should not occur
    # given the problem constraints and properties of total orders, but the logic
    # correctly handles it by printing -1), the strongest programmer
    # cannot be uniquely determined from the given information.
    print(-1)