def main():
    N, M = map(int, input().split())
    
    stand_flavors_masks = []
    for _ in range(N):
        s_line = input()
        mask = 0
        for j in range(M): # j is 0-indexed flavor, S_line[j] is for flavor j
            if s_line[j] == 'o':
                mask |= (1 << j)
        stand_flavors_masks.append(mask)
        
    target_mask = (1 << M) - 1
    
    # Initialize min_num_stands. N is a safe upper bound because visiting all N stands
    # is guaranteed to cover all flavors (as per problem constraints).
    min_num_stands = N 

    # Iterate through all possible subsets of stands.
    # A subset is represented by a bitmask `subset_config_mask` from 0 to (1 << N) - 1.
    # If the k-th bit of `subset_config_mask` is set, stand k is in the subset.
    for subset_config_mask in range(1 << N): # Iterate from 0 up to 2^N - 1
        current_flavors_achieved = 0
        num_stands_in_subset = 0
        
        # For the current subset `subset_config_mask`, calculate total flavors and count of stands
        for stand_idx in range(N): # stand_idx is stand index from 0 to N-1
            # Check if stand `stand_idx` is in the current subset `subset_config_mask`
            if (subset_config_mask >> stand_idx) & 1: 
                num_stands_in_subset += 1
                current_flavors_achieved |= stand_flavors_masks[stand_idx]
        
        # If all flavors are achieved with this subset
        if current_flavors_achieved == target_mask:
            min_num_stands = min(min_num_stands, num_stands_in_subset)
            
    print(min_num_stands)

if __name__ == '__main__':
    main()