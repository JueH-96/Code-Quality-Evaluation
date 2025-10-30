import sys

def solve():
    """
    Reads the problem input, solves it, and prints the output.
    """
    # Read the number of people, N.
    try:
        n_str = sys.stdin.readline()
        if not n_str: return
        n = int(n_str)
    except (IOError, ValueError):
        return

    # Read the sequence of people in the line, P.
    p_list = list(map(int, sys.stdin.readline().split()))

    # To efficiently answer the queries, we first build a map that allows us
    # to find the position of a person given their number.
    # Since person numbers are integers from 1 to N, a simple list (acting as a
    # direct-address table) is a very efficient choice for this map.
    # `positions[k]` will store the 1-based position of person `k`.
    # The list is of size N+1 to allow for 1-based indexing (person numbers 1 to N).
    positions = [0] * (n + 1)

    # Populate the positions map.
    # We iterate through the input list `p_list` with their 0-based indices.
    for i, person_number in enumerate(p_list):
        # The person `person_number` is at 0-based index `i`.
        # This corresponds to the (i+1)-th position in the line.
        positions[person_number] = i + 1

    # Read the number of queries, Q.
    try:
        q_str = sys.stdin.readline()
        if not q_str: return
        q = int(q_str)
    except (IOError, ValueError):
        return

    # Process each of the Q queries.
    for _ in range(q):
        # Read the two person numbers for the current query.
        a, b = map(int, sys.stdin.readline().split())
        
        # Retrieve the positions of person A and person B from our map.
        # This is a constant time O(1) lookup.
        position_of_a = positions[a]
        position_of_b = positions[b]
        
        # Compare the positions. The person with the smaller position number is
        # standing further to the front of the line.
        if position_of_a < position_of_b:
            print(a)
        else:
            print(b)

solve()