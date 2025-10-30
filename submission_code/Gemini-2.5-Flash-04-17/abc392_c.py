import sys

def solve():
    # Read input N
    N = int(sys.stdin.readline())

    # Read input P and convert to 0-indexed list
    # P_list[j] = person (0-indexed) that person j (0-indexed) stares at
    P_input = list(map(int, sys.stdin.readline().split()))
    P_list = [p - 1 for p in P_input]

    # Read input Q and convert to 0-indexed list
    # Q_list[j] = bib number (0-indexed) that person j (0-indexed) wears
    Q_input = list(map(int, sys.stdin.readline().split()))
    Q_list = [q - 1 for q in Q_input]

    # Create a mapping from bib number (0-indexed) to person (0-indexed)
    # bib_to_person[b] = person (0-indexed) wearing bib b (0-indexed)
    bib_to_person = [0] * N
    for j in range(N):
        # Person j (0-indexed) wears bib Q_list[j] (0-indexed)
        bib_to_person[Q_list[j]] = j

    # Calculate S_i for each target bib number i from 1 to N (1-indexed)
    # S_i is the bib number (1-indexed) of the person that
    # the person wearing bib i (1-indexed) is staring at.
    S_list = [0] * N

    # Iterate through target bib numbers i from 1 to N (1-indexed)
    for i in range(1, N + 1):
        # Target bib number (0-indexed)
        target_bib_0_indexed = i - 1

        # Find the person (0-indexed) wearing this bib number
        # This person is bib_to_person[target_bib_0_indexed]
        person_wearing_bib = bib_to_person[target_bib_0_indexed]

        # Find the person (0-indexed) that this person is staring at
        # This person is P_list[person_wearing_bib]
        person_stared_at = P_list[person_wearing_bib]

        # Find the bib number (0-indexed) of the person being stared at
        # This bib number is Q_list[person_stared_at]
        bib_of_stared_at = Q_list[person_stared_at]

        # The result S_i is this bib number converted back to 1-indexed
        S_i = bib_of_stared_at + 1

        # Store S_i at the corresponding index in S_list (index i-1)
        S_list[i - 1] = S_i

    # Print the results separated by spaces
    print(*S_list)

# Execute the solve function
solve()