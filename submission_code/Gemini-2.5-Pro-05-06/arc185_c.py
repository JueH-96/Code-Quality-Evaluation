import sys

def solve():
    N, X = map(int, sys.stdin.readline().split())
    A_list = list(map(int, sys.stdin.readline().split()))

    # Store elements as (value, original_index_0_based)
    B = []
    for i in range(N):
        B.append((A_list[i], i))

    # Sort B by value. If values are same, original index order doesn't matter for this step.
    B.sort()

    # Iterate B[p_outer] as one of the elements.
    # p_outer goes from 0 to N-3 because we need at least two other elements (left and right).
    for p_outer in range(N - 2):
        val_outer = B[p_outer][0]
        idx_outer = B[p_outer][1]
        
        target_sum_lr = X - val_outer
        
        # Optimization: Since A_l and A_r must be >= 1, their sum must be >= 2.
        # If target_sum_lr < 2, no valid pair A_l, A_r exists.
        # Since B is sorted by value, val_outer is non-decreasing.
        # If target_sum_lr becomes < 2 for current val_outer, it will be so for all subsequent larger val_outer.
        # So we can break the outer loop.
        if target_sum_lr < 2:
            break

        left = p_outer + 1
        right = N - 1

        while left < right:
            val_l = B[left][0]
            idx_l = B[left][1]
            val_r = B[right][0]
            idx_r = B[right][1]
            
            current_sum_lr = val_l + val_r

            if current_sum_lr == target_sum_lr:
                # Found three values (val_outer, val_l, val_r) that sum to X.
                # Their original 0-based indices are idx_outer, idx_l, idx_r.
                # These are guaranteed to be distinct because p_outer, left, right are distinct indices in B,
                # and B elements correspond to unique original positions.
                
                # Sort the original indices to satisfy the problem's i < j < k condition.
                indices_0_based = sorted([idx_outer, idx_l, idx_r])
                
                # Print 1-based indices.
                print(indices_0_based[0] + 1, indices_0_based[1] + 1, indices_0_based[2] + 1)
                return
            elif current_sum_lr < target_sum_lr:
                # Sum is too small, need larger values. Move left pointer.
                left += 1
            else: # current_sum_lr > target_sum_lr
                # Sum is too large, need smaller values. Move right pointer.
                right -= 1
                
    # If loops complete without finding a solution
    print("-1")

solve()