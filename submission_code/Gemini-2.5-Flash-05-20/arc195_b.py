import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    fixed_sums_set = set()
    s_lower_bound = 0 

    # Step 1: Collect information for S value determination
    # Iterate through initial A and B to find any fixed sum (A_i + B_i)
    # and determine the minimum possible S (s_lower_bound) required for elements to be non-negative.
    for i in range(N):
        if A[i] != -1 and B[i] != -1:
            fixed_sums_set.add(A[i] + B[i])
        
        # S must be at least as large as any existing non-negative A_i or B_i.
        if A[i] != -1:
            s_lower_bound = max(s_lower_bound, A[i])
        if B[i] != -1:
            s_lower_bound = max(s_lower_bound, B[i])

    # Step 2: Determine the definitive S value
    s_val = -1
    if len(fixed_sums_set) > 1:
        # If there are multiple (A_i, B_i) pairs where both are not -1,
        # and they sum to different values, it's impossible.
        print("No")
        return
    elif len(fixed_sums_set) == 1:
        # S is uniquely determined by a fixed (A_i, B_i) pair.
        s_val = fixed_sums_set.pop()
    else:
        # S is not determined by any (A_i, B_i) pair where both are non-negative.
        # In this case, we choose the smallest possible S that ensures all existing
        # non-negative A_i/B_i can be part of the sum.
        s_val = s_lower_bound
    
    # Step 3: Validate the determined S_val against lower bounds.
    # If the fixed S (s_candidate) is less than the maximum of existing A_i or B_i,
    # then it's impossible to make those elements non-negative while satisfying A_i+B_i=S.
    if s_val < s_lower_bound:
        print("No")
        return

    # Step 4: Construct the final A and B value lists based on s_val.
    final_A_values = []
    final_B_values = []

    for i in range(N):
        if A[i] != -1 and B[i] != -1:
            # Both A_i and B_i are fixed.
            # Their sum (A[i] + B[i]) must equal s_val. This consistency is already
            # implicitly handled by how s_val was determined (via fixed_sums_set).
            # If s_val came from fixed_sums_set, this condition is true.
            # If s_val came from s_lower_bound, but this pair exists, then fixed_sums_set was empty
            # and this branch implies a design flaw in s_val selection, which has been corrected.
            # An explicit check here is redundant but harmless:
            if A[i] + B[i] != s_val:
                print("No")
                return
            final_A_values.append(A[i])
            final_B_values.append(B[i])

        elif A[i] == -1 and B[i] != -1:
            # A_i needs to be determined as S - B_i.
            determined_A_i = s_val - B[i]
            if determined_A_i < 0:
                # Cannot make A_i non-negative.
                print("No")
                return
            final_A_values.append(determined_A_i)
            final_B_values.append(B[i])

        elif A[i] != -1 and B[i] == -1:
            # B_i needs to be determined as S - A_i.
            determined_B_i = s_val - A[i]
            if determined_B_i < 0:
                # Cannot make B_i non-negative.
                print("No")
                return
            final_A_values.append(A[i])
            final_B_values.append(determined_B_i)

        else: # A[i] == -1 and B[i] == -1
            # Both A_i and B_i can be chosen. We need A_i + B_i = S.
            # To maximize the chance of satisfying the sorted sum condition later,
            # we make A_i as small as possible (0) and B_i as large as possible (S).
            # If a solution exists with these extreme values, a solution exists with
            # any other valid choices for flexible slots (e.g., A_i=x, B_i=S-x for x in [0, S]).
            final_A_values.append(0)
            final_B_values.append(s_val)

    # Step 5: Sort and check the pairing condition.
    # We can rearrange A. This means the multiset of final A values must be able to
    # be paired with the multiset of final B values to sum to S.
    # This is equivalent to sorting one list ascending and the other descending,
    # and checking that their element-wise sum is S.
    final_A_values.sort()
    final_B_values.sort(reverse=True) # Sort B descending for pairwise sum check

    for i in range(N):
        if final_A_values[i] + final_B_values[i] != s_val:
            print("No")
            return

    print("Yes")

solve()