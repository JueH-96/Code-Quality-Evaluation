import sys

def solve():
    # Read N and M from the first line of input.
    # N: number of competitive programmers (1 to N)
    # M: number of pieces of information (superiority relations)
    N, M = map(int, sys.stdin.readline().split())

    # Initialize a 2D array (adjacency matrix) to represent superiority relations.
    # `is_stronger[i][j]` will be True if person `i` is known to be stronger than person `j`.
    # We use N+1 size for both dimensions to allow for 1-based indexing directly.
    is_stronger = [[False for _ in range(N + 1)] for _ in range(N + 1)]

    # Initialize diagonal elements to True.
    # This signifies that a person is "stronger than themselves" (reflexive property).
    # While not strictly necessary for the problem's definition of the strongest programmer,
    # it is a standard setup for the Floyd-Warshall algorithm for reachability and simplifies the logic.
    for i in range(1, N + 1):
        is_stronger[i][i] = True

    # Read the M given superiority relations and update the matrix.
    # If person A is stronger than person B, set is_stronger[A][B] to True.
    for _ in range(M):
        A, B = map(int, sys.stdin.readline().split())
        is_stronger[A][B] = True

    # Compute the transitive closure using the Floyd-Warshall algorithm.
    # This step infers all direct and indirect superiority relations based on transitivity.
    # The 'k' loop represents the intermediate programmer in a path i -> k -> j.
    for k in range(1, N + 1):
        # The 'i' loop represents the starting programmer.
        for i in range(1, N + 1):
            # The 'j' loop represents the ending programmer.
            for j in range(1, N + 1):
                # If 'i' is stronger than 'k' AND 'k' is stronger than 'j',
                # then by transitivity, 'i' is also stronger than 'j'.
                # Update is_stronger[i][j] if a new superiority path is found.
                is_stronger[i][j] = is_stronger[i][j] or (is_stronger[i][k] and is_stronger[k][j])

    # Identify potential strongest programmers.
    strongest_candidates = []
    # Iterate through each programmer `p` from 1 to N.
    for p in range(1, N + 1):
        # Count how many distinct programmers `p` is proven to be stronger than.
        beats_count = 0
        # Compare `p` with every other programmer `other_p`.
        for other_p in range(1, N + 1):
            if p == other_p:
                continue # A person cannot be stronger than themselves in this context.

            # If person `p` is proven to be stronger than `other_p` (via direct or inferred relation).
            if is_stronger[p][other_p]:
                beats_count += 1
        
        # A programmer is a candidate for the strongest if they are proven to be
        # stronger than all N-1 other programmers.
        if beats_count == N - 1:
            strongest_candidates.append(p)

    # Determine the final output based on the number of strongest candidates found.
    if len(strongest_candidates) == 1:
        # If exactly one programmer meets the criteria, print their number.
        print(strongest_candidates[0])
    else:
        # If zero or more than one programmer meets the criteria,
        # the strongest programmer cannot be uniquely determined. Print -1.
        print(-1)

# Call the solve function to execute the program logic.
solve()