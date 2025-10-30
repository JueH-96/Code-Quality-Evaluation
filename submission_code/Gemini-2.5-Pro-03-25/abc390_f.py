# YOUR CODE HERE
import sys

# Function to read input faster
readline = sys.stdin.readline

def solve():
    N = int(readline())
    # Constraints state N >= 1, so N=0 check is not strictly necessary but harmless for robustness
    if N == 0: 
        print(0)
        return
        
    A = list(map(int, readline().split()))

    # Store 1-based indices for each value. The size is N+1 because values can be up to N.
    # pos[k] will contain a sorted list of indices i such that A[i-1] = k.
    pos = [[] for _ in range(N + 1)]
    for i in range(N):
        val = A[i]
        # Constraints state 1 <= A_i <= N.
        if 1 <= val <= N:
             pos[val].append(i + 1) # Store 1-based index

    # Helper function to calculate the total count of subsegments that are fully contained
    # within intervals where elements corresponding to 'indices' are absent.
    # Mathematically, this calculates Sum_{intervals I disjoint from indices} |I|*(|I|+1)/2
    # 'indices' is a sorted list of positions where a certain value (or set of values) appears.
    def calculate_sum_M(indices, N):
        current_sum = 0
        p_prev = 0 # Start from virtual index 0 before the first element
        for p_curr in indices:
            # Consider the segment of indices strictly between p_prev and p_curr.
            # Indices are [p_prev + 1, ..., p_curr - 1]
            # Length of this segment is (p_curr - 1) - (p_prev + 1) + 1 = p_curr - p_prev - 1
            M = p_curr - p_prev - 1
            if M > 0:
                # The number of subsegments within a segment of length M is M * (M + 1) // 2
                current_sum += M * (M + 1) // 2
            # Update p_prev for the next iteration
            p_prev = p_curr
        
        # After iterating through all indices, consider the last segment.
        # This segment is from p_prev + 1 to N. Indices are [p_prev + 1, ..., N].
        # Length of this segment is N - (p_prev + 1) + 1 = N - p_prev
        M = N - p_prev
        if M > 0:
            current_sum += M * (M + 1) // 2
        return current_sum

    # Helper function to merge two sorted lists of indices and return a unique sorted list
    # This is used to find the union of positions S_v U S_{v-1}
    def merge_sorted_unique(list1, list2):
        res = []
        i = 0
        j = 0
        # Use a variable to track the last element added to avoid duplicates from merge.
        # Indices are positive integers >= 1. Use 0 as initial state safe value.
        last_added = 0 
        
        while i < len(list1) and j < len(list2):
            if list1[i] < list2[j]:
                val = list1[i]
                # Check if this value is different from last added element
                if val != last_added:
                    res.append(val)
                    last_added = val
                i += 1
            elif list1[i] > list2[j]:
                val = list2[j]
                if val != last_added:
                    res.append(val)
                    last_added = val
                j += 1
            else: # list1[i] == list2[j]
                 val = list1[i]
                 if val != last_added:
                     res.append(val)
                     last_added = val
                 # Since elements are equal, advance both pointers
                 i += 1
                 j += 1
        
        # Append remaining elements from list1
        while i < len(list1):
            val = list1[i]
            if val != last_added:
                 res.append(val)
                 last_added = val
            i += 1
        # Append remaining elements from list2
        while j < len(list2):
            val = list2[j]
            if val != last_added:
                res.append(val)
                last_added = val
            j += 1
        return res

    total_f_sum = 0

    # Total number of possible pairs (L, R) with 1 <= L <= R <= N
    total_pairs = N * (N + 1) // 2

    # Calculate contribution C(v) for each value v from 1 to N
    # C(v) = Count of pairs (L, R) such that v is in {A_L, ..., A_R} and v-1 is not.
    # This counts pairs (L, R) where v starts a new contiguous block of values.

    # Case v = 1
    # For v=1, the condition 'v-1 not in {A_L, ..., A_R}' is always true since A_i >= 1 (and thus 0 is never present).
    # C(1) = Count of pairs (L, R) such that 1 is in {A_L, ..., A_R}.
    S1_indices = pos[1]
    # Count pairs disjoint from S1 using helper function.
    count_disjoint_1 = calculate_sum_M(S1_indices, N)
    # C(1) = Total pairs - Pairs disjoint from S1
    C1 = total_pairs - count_disjoint_1
    total_f_sum += C1

    # Case v > 1
    for v in range(2, N + 1):
        Sv_indices = pos[v]
        # If value v does not appear in A, it cannot start a block, so its contribution C(v) is 0.
        if not Sv_indices:
            continue

        # Get indices where v-1 appears. S_{v-1}.
        Sv_minus_1_indices = pos[v-1]
        
        # Calculate number of pairs (L, R) disjoint from S_{v-1}.
        # These are the pairs where v-1 does NOT appear.
        count_disjoint_v_minus_1 = calculate_sum_M(Sv_minus_1_indices, N)
        
        # Compute the union of indices for v and v-1. Let this set be P = S_v U S_{v-1}.
        P_indices = merge_sorted_unique(Sv_indices, Sv_minus_1_indices)
        
        # Calculate number of pairs (L, R) disjoint from P.
        # These are pairs where NEITHER v NOR v-1 appear.
        count_disjoint_v_and_v_minus_1 = calculate_sum_M(P_indices, N)
        
        # The contribution C(v) is the count of pairs (L,R) containing v but not v-1.
        # This can be derived using the principle of inclusion-exclusion or set theory:
        # |{(L,R) | v \in V_{L,R} \land v-1 
otin V_{L,R}}|
        # = |{(L,R) | v-1 
otin V_{L,R}}| - |{(L,R) | v 
otin V_{L,R} \land v-1 
otin V_{L,R}}|
        # = (Pairs disjoint from S_{v-1}) - (Pairs disjoint from P)
        Cv = count_disjoint_v_minus_1 - count_disjoint_v_and_v_minus_1
        total_f_sum += Cv

    print(total_f_sum)

solve()