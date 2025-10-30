import sys

def solve():
    N = int(sys.stdin.readline())
    
    # P_val_list[i] is the 1-indexed person ID that "Person i+1" (from problem statement, 
    # which is 0-indexed person `i` in our internal logic) stares at.
    P_val_list = list(map(int, sys.stdin.readline().split()))
    
    # Q_val_list[i] is the 1-indexed bib number on "Person i+1" (0-indexed person `i`).
    Q_val_list = list(map(int, sys.stdin.readline().split()))

    # Internal representation:
    # - Person IDs will be 0-indexed (0 to N-1).
    # - Bib numbers will remain 1-indexed (1 to N) as they are values, not indices directly.

    # Map 1: person_stares_at_target_map
    #   Input: 0-indexed ID of staring person.
    #   Output: 0-indexed ID of target person.
    #   Logic: 0-indexed person `j` stares at 1-indexed person `P_val_list[j]`.
    #          So, target person's 0-indexed ID is `P_val_list[j] - 1`.
    person_stares_at_target_map = [p_val - 1 for p_val in P_val_list]

    # Map 2: bib_on_person_map
    #   Input: 0-indexed ID of a person.
    #   Output: 1-indexed bib number they wear.
    #   Logic: 0-indexed person `j` wears bib `Q_val_list[j]`.
    #   This map is essentially Q_val_list itself.
    bib_on_person_map = Q_val_list 

    # Map 3: person_wearing_bib_map
    #   Input: 1-indexed bib number.
    #   Output: 0-indexed ID of person wearing that bib.
    #   Bib numbers are 1 to N, so array size N+1. Index 0 is unused.
    person_wearing_bib_map = [0] * (N + 1) 
    for person_idx_0_based in range(N):
        bib_num_on_this_person = Q_val_list[person_idx_0_based] # This is 1-indexed
        person_wearing_bib_map[bib_num_on_this_person] = person_idx_0_based

    # ans_list[k] will store S_{k+1} (the result for query_bib_num = k+1).
    ans_list = [0] * N 

    # Iterate for each bib number `b` from 1 to N (as per problem: S_1, S_2, ..., S_N).
    for query_bib_num_1_based in range(1, N + 1):
        # Step 1: Find who is wearing `query_bib_num_1_based`.
        # This gives person_A's 0-indexed ID.
        person_A_idx_0_based = person_wearing_bib_map[query_bib_num_1_based]
        
        # Step 2: Find who person_A is staring at.
        # This gives person_B's 0-indexed ID.
        person_B_idx_0_based = person_stares_at_target_map[person_A_idx_0_based]
        
        # Step 3: Find the bib number on person_B.
        # This is the result for `query_bib_num_1_based`.
        bib_of_person_B_1_based = bib_on_person_map[person_B_idx_0_based]
        
        # Store the result. `query_bib_num_1_based` is 1-indexed (1 to N).
        # `ans_list` is 0-indexed, so store S_b at index b-1.
        ans_list[query_bib_num_1_based - 1] = bib_of_person_B_1_based
        
    # Print results S_1, S_2, ..., S_N separated by spaces.
    sys.stdout.write(" ".join(map(str, ans_list)) + "
")

if __name__ == '__main__':
    solve()