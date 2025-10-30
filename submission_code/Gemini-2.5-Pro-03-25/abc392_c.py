# YOUR CODE HERE
import sys

def solve():
    # Read the number of people, N.
    n = int(sys.stdin.readline())

    # Read the list P. P[k] is the 1-based index of the person that person k+1 (0-based index k) is staring at.
    p = list(map(int, sys.stdin.readline().split()))

    # Read the list Q. Q[k] is the bib number worn by person k+1 (0-based index k).
    q = list(map(int, sys.stdin.readline().split()))

    # Convert the P list to use 0-based indexing for the stared-at person.
    # p_0based[k] will store the 0-based index of the person stared at by person k (0-based index k).
    p_0based = [pi - 1 for pi in p]

    # Let's think in terms of 0-based indices for people (0 to n-1).
    # Person k (index k) corresponds to the (k+1)-th person in the 1-based input description.
    # The bib number of person k (index k) is q[k].
    # Person k (index k) stares at the person with index p_0based[k].

    # We need to compute S_i for each bib number i (from 1 to N).
    # S_i is the bib number of the person that the person wearing bib i is staring at.

    # Step 1: For each person k, find the bib number of the person they are staring at.
    # bib_of_stared_at[k] will store the bib number of the person stared at by person k (index k).
    bib_of_stared_at = [0] * n
    for k in range(n):
        # Person k stares at the person with index p_0based[k].
        stared_at_person_index = p_0based[k]

        # Find the bib number of the person at index stared_at_person_index.
        # The bib number of person j (index j) is q[j].
        bib_of_stared_person = q[stared_at_person_index]

        # Store this bib number, associated with the staring person k.
        bib_of_stared_at[k] = bib_of_stared_person

    # Step 2: Create the final result array `result_s`.
    # result_s[i-1] will store S_i.
    result_s = [0] * n

    # Iterate through each person k from 0 to n-1.
    for k in range(n):
        # Person k (index k) wears bib q[k].
        person_k_bib = q[k]

        # Person k (index k) stares at someone wearing bib bib_of_stared_at[k].
        bib_stared_at_by_k = bib_of_stared_at[k]

        # We need to find S_i, the bib number of the person stared at by the person wearing bib i.
        # Let i = person_k_bib. The person wearing bib i is person k.
        # Person k stares at someone with bib number bib_stared_at_by_k.
        # Therefore, S_i = S_{person_k_bib} = bib_stared_at_by_k.

        # Store this result in the `result_s` array. Since bib numbers are 1-based (1 to N)
        # and array indices are 0-based (0 to N-1), we store S_i at index i-1.
        # So, store S_{person_k_bib} at index (person_k_bib - 1).
        result_s[person_k_bib - 1] = bib_stared_at_by_k

    # Print the final results S_1, S_2, ..., S_N, separated by spaces.
    # Using ' '.join(map(str, ...)) is generally efficient for printing lists of numbers.
    print(' '.join(map(str, result_s)))

# Run the solution function
solve()

# END OF YOUR CODE HERE