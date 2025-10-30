# YOUR CODE HERE
import sys

def solve():
    """
    Solves the problem by mapping relationships between people and bibs.
    """
    # Read problem size from the first line of stdin
    try:
        N = int(sys.stdin.readline())
        if N == 0: return # Handle empty input case
        # P[i] is the person (1-based) that person i+1 stares at
        P = list(map(int, sys.stdin.readline().split()))
        # Q[i] is the bib number (1-based) on person i+1
        Q = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        # Gracefully exit on malformed input or end-of-file
        return

    # To solve this efficiently, we need a way to find the person wearing a given bib.
    # We create a map from a bib number to the 0-based index of the person wearing it.
    # `person_idx_from_bib[b]` will store the 0-based index of the person wearing bib `b`.
    # The array size is N+1 to accommodate 1-based bib numbers from 1 to N.
    person_idx_from_bib = [0] * (N + 1)
    for i in range(N):
        # Person with 0-based index `i` (i.e., Person i+1) wears bib `Q[i]`.
        bib_num = Q[i]
        person_idx_from_bib[bib_num] = i

    # The problem asks for a sequence S_1, S_2, ..., S_N.
    # S_i is the bib of the person stared at by the person wearing bib i.
    result_s = [0] * N

    # Iterate through each bib number `i` from 1 to N to find the corresponding S_i.
    for bib_num_i in range(1, N + 1):
        # Step 1: Find the person wearing bib `bib_num_i`.
        # Their 0-based index is `staring_person_idx`.
        staring_person_idx = person_idx_from_bib[bib_num_i]
        
        # Step 2: Find who this person stares at.
        # Person `staring_person_idx` (0-based) stares at person with 1-based number `P[staring_person_idx]`.
        # The 0-based index of the person being stared at is `stared_at_person_idx`.
        stared_at_person_idx = P[staring_person_idx] - 1
        
        # Step 3: Find the bib number of the person being stared at.
        # The person with 0-based index `stared_at_person_idx` wears bib `Q[stared_at_person_idx]`.
        final_bib = Q[stared_at_person_idx]
        
        # The result for bib `bib_num_i` is `final_bib`.
        # We store it in our 0-indexed result list at the appropriate position.
        result_s[bib_num_i - 1] = final_bib

    # Print the final result list with elements separated by spaces.
    # The `*` operator unpacks the list into arguments for the print function.
    print(*result_s)

solve()