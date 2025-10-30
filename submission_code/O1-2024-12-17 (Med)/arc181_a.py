def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])
    idx = 1
    
    # A helper function to check if a given permutation is already the identity.
    def is_identity(permutation):
        return all(permutation[i] == i+1 for i in range(len(permutation)))
    
    # We'll solve each test case in O(N) by using the fact that the answer is always 0, 1, or 2:
    # 1) If already sorted => 0
    # 2) Else check if there is a single "pivot" k that fixes it in one operation => 1
    # 3) Otherwise => 2
    
    # Condition for 1-operation fix with pivot k (1-based):
    #  - the pivot's element stays the same (so it will remain P[k]), 
    #  - the left side [1..k-1] gets sorted and should become {1,2,...,k-1},
    #  - the right side [k+1..N] gets sorted and should become {k+1,...,N}.
    # This requires:
    #  - P[k] == k,
    #  - the subarray [1..k-1] is exactly the set {1..k-1},
    #  - the subarray [k+1..N] is exactly the set {k+1..N}.
    
    # We can precompute:
    #   prefixCheck[i] = True if P[1..i] is exactly {1..i}
    #   suffixCheck[i] = True if P[i..N] is exactly {i..N}
    #
    # Then for any k:
    #   if prefixCheck[k-1] and P[k] = k and suffixCheck[k+1] => we can do it in 1 operation.
    #
    # If none such k works (and it's not already identity), answer is 2.
    
    out = []
    for _ in range(t):
        N = int(input_data[idx]); idx += 1
        P = list(map(int, input_data[idx:idx+N]))
        idx += N
        
        # Check if already identity:
        if is_identity(P):
            out.append("0")
            continue
        
        # Build prefixCheck and suffixCheck (1-based indexing).
        prefixCheck = [False]*(N+1)
        # prefixCheck[i] = True if subarray P[1..i] == {1..i}.
        # We'll track running sum, min, max to confirm it is exactly {1..i}.
        run_sum = 0
        run_max = 0
        run_min = N+1
        for i in range(1, N+1):
            val = P[i-1]
            run_sum += val
            if val > run_max: 
                run_max = val
            if val < run_min:
                run_min = val
            length = i
            # sum of {1..i} = i*(i+1)//2
            if run_max == i and run_min == 1 and run_sum == (i*(i+1))//2:
                prefixCheck[i] = True
        
        suffixCheck = [False]*(N+2)
        # suffixCheck[i] = True if subarray P[i..N] == {i..N}.
        # We'll track running sum, min, max from the right
        total_sum = N*(N+1)//2
        run_sum = 0
        run_max = 0
        run_min = N+1
        for i in range(N, 0, -1):
            val = P[i-1]
            run_sum += val
            if val > run_max:
                run_max = val
            if val < run_min:
                run_min = val
            length = N - i + 1
            # sum of {i..N} = sum(1..N) - sum(1..i-1) = total_sum - (i-1)*i//2
            needed_sum = total_sum - ((i-1)*i)//2
            if run_max == N and run_min == i and run_sum == needed_sum:
                suffixCheck[i] = True
        
        # We'll also define prefixCheck[0] = True (empty set) and suffixCheck[N+1] = True (empty set) for boundary.
        # Instead of storing them, we can directly treat them as True:
        
        can_do_in_one = False
        for k in range(1, N+1):
            # prefix part is k-1, suffix part is k+1..N
            # boundary check:
            left_ok = (k == 1) or prefixCheck[k-1]
            right_ok = (k == N) or suffixCheck[k+1]
            if left_ok and (P[k-1] == k) and right_ok:
                can_do_in_one = True
                break
        
        if can_do_in_one:
            out.append("1")
        else:
            out.append("2")
    
    print("
".join(out))

# Do not forget to call main()!
main()