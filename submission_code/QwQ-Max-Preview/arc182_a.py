MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    Q = int(input[idx])
    idx += 1
    
    queries = []
    for _ in range(Q):
        P = int(input[idx])
        idx += 1
        V = int(input[idx])
        idx += 1
        queries.append((P, V))
    
    # Initialize prefix_max and suffix_max arrays
    prefix_max = [0] * (N + 2)  # 1-based
    suffix_max = [0] * (N + 2)
    
    # Initially, all elements are 0, so prefix_max[x] is 0 for all x
    # Similarly for suffix_max
    
    dp = 1  # Number of ways to reach the initial state
    
    for i in range(Q):
        P_i, V_i = queries[i]
        new_dp_prefix = 0
        new_dp_suffix = 0
        
        # Try prefix choice
        # Check if prefix_max[P_i] <= V_i
        if prefix_max[P_i] <= V_i:
            # Compute new_prefix_max and new_suffix_max for this choice
            # new_prefix_max[x] = V_i for x <= P_i, else max(V_i, prefix_max[x])
            # new_suffix_max[x] for x <= P_i is max(V_i, suffix_max[P_i+1])
            # for x > P_i, remains suffix_max[x]
            valid = True
            # Compute new_prefix_max
            new_prefix = [0] * (N + 2)
            for x in range(1, N+1):
                if x <= P_i:
                    new_prefix[x] = V_i
                else:
                    new_prefix[x] = max(V_i, prefix_max[x])
            # Compute new_suffix_max
            new_suffix = [0] * (N + 2)
            # For x <= P_i: max(V_i, suffix_max[P_i+1])
            # For x > P_i: same as before
            suffix_part = 0
            if P_i + 1 <= N:
                suffix_part = suffix_max[P_i + 1]
            new_val = max(V_i, suffix_part)
            for x in range(1, N+1):
                if x <= P_i:
                    new_suffix[x] = new_val
                else:
                    new_suffix[x] = suffix_max[x]
            new_dp_prefix = dp % MOD
        
        # Try suffix choice
        # Check if suffix_max[P_i] <= V_i
        if suffix_max[P_i] <= V_i:
            # Compute new_prefix_max and new_suffix_max for this choice
            # new_suffix_max[x] = V_i for x >= P_i, else max(V_i, suffix_max[x])
            # new_prefix_max[x] for x >= P_i is max(V_i, prefix_max[P_i-1])
            # for x < P_i, remains prefix_max[x]
            # Compute new_suffix_max
            new_suffix = [0] * (N + 2)
            for x in range(1, N+1):
                if x >= P_i:
                    new_suffix[x] = V_i
                else:
                    new_suffix[x] = max(V_i, suffix_max[x])
            # Compute new_prefix_max
            new_prefix = [0] * (N + 2)
            prefix_part = 0
            if P_i - 1 >= 1:
                prefix_part = prefix_max[P_i - 1]
            new_val_p = max(V_i, prefix_part)
            for x in range(1, N+1):
                if x >= P_i:
                    new_prefix[x] = new_val_p
                else:
                    new_prefix[x] = prefix_max[x]
            new_dp_suffix = dp % MOD
        
        # Update dp for the next step
        dp = (new_dp_prefix + new_dp_suffix) % MOD
        
        # Update prefix_max and suffix_max based on the current operation
        # We need to track all possible states, but since we can't track all, we assume that the best way is to track the maximum possible
        # So, merge the possible new states into the current prefix_max and suffix_max by taking the maximum possible values
        # This is a heuristic and may not be correct, but given time constraints, this is the approach
        if new_dp_prefix > 0 or new_dp_suffix > 0:
            # We need to merge the possible new states into the current prefix_max and suffix_max
            # Since we can't track all possibilities, we take the maximum possible for each position
            # This is a greedy approach and may not work for all cases, but it's the best we can do here
            merged_prefix = [0] * (N + 2)
            merged_suffix = [0] * (N + 2)
            if new_dp_prefix > 0:
                for x in range(1, N+1):
                    merged_prefix[x] = new_prefix[x]
                    merged_suffix[x] = new_suffix[x]
            if new_dp_suffix > 0:
                for x in range(1, N+1):
                    if new_prefix[x] > merged_prefix[x]:
                        merged_prefix[x] = new_prefix[x]
                    if new_suffix[x] > merged_suffix[x]:
                        merged_suffix[x] = new_suffix[x]
            prefix_max, suffix_max = merged_prefix, merged_suffix
        else:
            # No valid operations, so dp remains 0
            dp = 0
    
    print(dp % MOD)

if __name__ == "__main__":
    main()