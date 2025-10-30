import sys

# The problem involves mapping relationships between people and bibs.
# We are given who each person looks at (P) and what bib each person wears (Q).
# We need to find, for each bib number i, the bib number of the person that the person wearing bib i is staring at.
# This requires chaining the relationships: bib number -> person -> person they look at -> their bib number.

def solve():
    # Read the number of people
    N = int(sys.stdin.readline())

    # Read the list P: P[i-1] is the person (1-indexed) that person i (1-indexed) is staring at.
    # We store P as a 0-indexed list of length N. P[i] holds the value P_{i+1}.
    P = list(map(int, sys.stdin.readline().split()))

    # Read the list Q: Q[i-1] is the number (1-indexed) on the bib of person i (1-indexed).
    # We store Q as a 0-indexed list of length N. Q[i] holds the value Q_{i+1}.
    Q = list(map(int, sys.stdin.readline().split()))

    # To go from a bib number to the person wearing it, we need the inverse mapping of Q.
    # Create a mapping from bib number (1 to N) to the person number (1 to N) who wears it.
    # who_wears_bib[bib_num] = person_num (both 1-indexed)
    # We use a list of size N+1 for 1-based indexing convenience.
    who_wears_bib = [0] * (N + 1)
    # Iterate through person indices (0 to N-1). Person i+1 is the current person.
    for i in range(N):
        # Person i+1 wears bib Q[i].
        # So, the person wearing bib Q[i] is i+1.
        who_wears_bib[Q[i]] = i + 1

    # To find who a person looks at, we use the P list.
    # Create a mapping from person number (1 to N) to the person number (1 to N) they look at.
    # who_person_i_looks_at[person_num] = looked_at_person_num (both 1-indexed)
    # We use a list of size N+1 for 1-based indexing convenience.
    who_person_i_looks_at = [0] * (N + 1)
    # Iterate through person indices (0 to N-1). Person i+1 is the current person.
    for i in range(N):
        # Person i+1 looks at person P[i].
        # So, the person person i+1 looks at is P[i].
        who_person_i_looks_at[i + 1] = P[i]

    # The result list S will store S_1, S_2, ..., S_N.
    # S[k-1] will be the bib number that the person wearing bib k is staring at.
    # This is a 0-indexed list of size N.
    S = [0] * N

    # We need to find the answer for each bib number from 1 to N.
    # Iterate through the target bib numbers (the bib numbers for which we need the answer)
    for target_bib_num in range(1, N + 1):
        # Step 1: Find the person who is wearing this target bib number.
        # Use the who_wears_bib map. This gives us a 1-indexed person number.
        person_wearing_current_bib = who_wears_bib[target_bib_num]

        # Step 2: Find the person that person_wearing_current_bib is looking at.
        # Use the who_person_i_looks_at map. This gives us a 1-indexed person number.
        person_they_look_at = who_person_i_looks_at[person_wearing_current_bib]

        # Step 3: Find the bib number of person_they_look_at.
        # The bib number of person k (1-indexed) is Q_k, which is stored at Q[k-1] in our 0-indexed list Q.
        # So, the bib number of person_they_look_at is Q[person_they_look_at - 1].
        bib_of_looked_at_person = Q[person_they_look_at - 1] # This is a 1-indexed bib number

        # Store the result. The result for target_bib_num (1-indexed)
        # goes into the result list S at index target_bib_num - 1 (0-indexed).
        S[target_bib_num - 1] = bib_of_looked_at_person

    # Print the resulting bib numbers separated by spaces.
    # The '*' operator unpacks the list S into arguments for the print function.
    print(*S)

# Execute the main solve function.
solve()