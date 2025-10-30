import collections

def solve():
    N = int(input())
    A_orig = list(map(int, input().split()))
    B_orig = list(map(int, input().split()))

    s_star = 0
    
    a_fixed_values = []
    num_a_minus_one = 0
    for x in A_orig:
        if x == -1:
            num_a_minus_one += 1
        else:
            a_fixed_values.append(x)
            s_star = max(s_star, x)

    # b_fixed_values_for_iteration will store the B_j != -1 values for iteration
    b_fixed_values_for_iteration = [] 
    for x in B_orig:
        if x != -1:
            b_fixed_values_for_iteration.append(x)
            s_star = max(s_star, x)

    # Create frequency map of fixed A values
    counts_a_map = collections.Counter(a_fixed_values)

    # For each B_j != -1, we need an A_value = S_star - B_j
    # Prefer using a fixed A_k, otherwise use an A_i = -1 wildcard.
    for b_val in b_fixed_values_for_iteration:
        # By construction of s_star, b_val <= s_star, so target_a_req is guaranteed >= 0.
        target_a_req = s_star - b_val
        
        if counts_a_map.get(target_a_req, 0) > 0:
            counts_a_map[target_a_req] -= 1
        elif num_a_minus_one > 0:
            num_a_minus_one -= 1
        else:
            # Not enough fixed A values matching target_a_req, and no A-wildcards left.
            print("No")
            return
            
    # If the loop completes, it means all fixed B values have been satisfied.
    # The remaining fixed A values and A-wildcards can be shown to correctly pair up
    # with B-wildcards due to an underlying counting identity.
    # Thus, no further checks are needed.
    
    print("Yes")

solve()