import sys

def solve():
    N = int(sys.stdin.readline())
    
    # Read P values. P_i is who person i is staring at.
    # P_list[i] means person i is staring at P_list[i].
    # We use 1-based indexing for person IDs and bib numbers to match problem statement.
    # So P_list will have N+1 elements, with index 0 unused.
    # We prepend a 0 to the list created by map to achieve 1-based indexing.
    P_list = [0] + list(map(int, sys.stdin.readline().split()))
    
    # Read Q values. Q_i is the bib number of person i.
    # Q_list[i] means person i is wearing bib Q_list[i].
    # Similarly, use 1-based indexing for Q_list.
    Q_list = [0] + list(map(int, sys.stdin.readline().split()))

    # Create an inverse mapping for Q:
    # person_by_bib[bib_number] will store the person ID (1-indexed) who wears that bib.
    person_by_bib = [0] * (N + 1)
    for person_id in range(1, N + 1):
        bib_number = Q_list[person_id]
        person_by_bib[bib_number] = person_id

    # S_list will store the results: S_k is the bib number of the person
    # that the person wearing bib k is staring at.
    # S_list will also have N+1 elements, with index 0 unused.
    S_list = [0] * (N + 1)

    # Calculate S_k for each bib number k from 1 to N.
    for k in range(1, N + 1):
        # Step 1: Find the person who is wearing bib k.
        # This person's ID (1-indexed) is stored in 'person_wearing_bib_k'.
        person_wearing_bib_k = person_by_bib[k]

        # Step 2: Find who this person ('person_wearing_bib_k') is staring at.
        # This person's ID (1-indexed) is stored in 'person_being_stared_at'.
        person_being_stared_at = P_list[person_wearing_bib_k]

        # Step 3: Find the bib number of the person they are staring at.
        # This is the value for S_k.
        bib_of_person_being_stared_at = Q_list[person_being_stared_at]
        
        S_list[k] = bib_of_person_being_stared_at

    # Print the results S_1, S_2, ..., S_N in order, separated by a single space.
    # We print elements from index 1 to N, excluding the unused 0th element.
    sys.stdout.write(" ".join(map(str, S_list[1:])) + "
")

# Call the solve function to run the program.
solve()