class Solution:
  def maxProduct(self, nums: list[int], k: int, limit: int) -> int:
    n = len(nums)

    # Maximum theoretical absolute value of an alternating sum.
    # If all nums[i] were 12, an alternating sum could be up to N * 12.
    # (e.g. [12] sum 12, [12, -12] sum 24 - no, this is not alternating sum def.)
    # Alt sum: S_0 - S_1 + S_2 ... Max for [12,0,12,0,...] is (N/2)*12 = 900 for N=150.
    # Using N*12 = 1800 as a safe upper bound for any sum magnitude.
    MAX_ABS_SUM_BOUND = n * 12 
    
    if abs(k) > MAX_ABS_SUM_BOUND: # k is unreachable if outside this broader bound
        return -1

    SUM_OFFSET = MAX_ABS_SUM_BOUND
    SUM_RANGE_SIZE = 2 * MAX_ABS_SUM_BOUND + 1

    # dp[sum_idx][parity] = max_product
    # parity: 0 for even length, 1 for odd length
    dp = [[-1] * 2 for _ in range(SUM_RANGE_SIZE)]
    
    # reached_non_empty[sum_idx][parity] = True if state was achieved via a non-empty subsequence
    reached_non_empty = [[False] * 2 for _ in range(SUM_RANGE_SIZE)]

    # Base case: empty subsequence (sum=0, len=0 (even), product=1)
    dp[SUM_OFFSET][0] = 1
    # reached_non_empty[SUM_OFFSET][0] is False for the empty subsequence

    for x_val in nums:
        new_dp = [row[:] for row in dp]
        new_rne = [row[:] for row in reached_non_empty]

        for current_sum_idx in range(SUM_RANGE_SIZE):
            for current_len_parity in range(2): # 0 for even, 1 for odd
                
                prev_prod = dp[current_sum_idx][current_len_parity]
                if prev_prod == -1: # This state (current_sum_idx, current_len_parity) was not reachable
                    continue

                # --- Option: Include x_val ---
                product_if_picked = -1 # Default if calculation fails (e.g. > limit)

                if x_val == 0:
                    product_if_picked = 0
                elif prev_prod == 0: # Current subsequence product is 0, and x_val is not 0
                    product_if_picked = 0
                else: # prev_prod > 0 and x_val > 0 (prev_prod starts at 1 for empty, nums[i]>=0)
                    # Check for product exceeding limit, carefully to avoid overflow if limit is huge
                    # (Here limit is small, direct multiplication is fine)
                    if x_val > 0 and prev_prod > limit / x_val : 
                        product_if_picked = limit + 1 # Mark as too large
                    else:
                        product_if_picked = prev_prod * x_val
                
                if product_if_picked > limit:
                    continue # This path leads to product > limit

                current_sum_val = current_sum_idx - SUM_OFFSET
                new_sum_val = 0
                new_len_parity = 0

                if current_len_parity == 0: # Current subsequence has even length
                    new_sum_val = current_sum_val + x_val # x_val is at an even pos in new subseq
                    new_len_parity = 1 # Length becomes odd
                else: # Current subsequence has odd length
                    new_sum_val = current_sum_val - x_val # x_val is at an odd pos in new subseq
                    new_len_parity = 0 # Length becomes even
                
                # new_sum_val must be within [-MAX_ABS_SUM_BOUND, MAX_ABS_SUM_BOUND]
                # This should hold due to how MAX_ABS_SUM_BOUND is defined.
                # If it were to go out of bounds, it means abs(k) > MAX_ABS_SUM_BOUND would have caught it.
                new_sum_idx = new_sum_val + SUM_OFFSET

                # Update new_dp and new_rne for the state (new_sum_idx, new_len_parity)
                # This path (picking x_val) results in a non-empty sequence.
                
                # Value in new_dp for (new_sum_idx, new_len_parity) comes from *not* picking x_val
                val_from_not_picking = new_dp[new_sum_idx][new_len_parity]
                rne_from_not_picking = new_rne[new_sum_idx][new_len_parity]

                if product_if_picked > val_from_not_picking:
                    new_dp[new_sum_idx][new_len_parity] = product_if_picked
                    new_rne[new_sum_idx][new_len_parity] = True # Path "pick x_val" is non-empty
                elif product_if_picked == val_from_not_picking:
                    if val_from_not_picking != -1: # If the state was already reachable
                        # The state is reachable by "not picking x_val" (non-empty status: rne_from_not_picking)
                        # AND by "picking x_val" (non-empty status: True).
                        # So, combined non-empty status is (rne_from_not_picking OR True) = True.
                        new_rne[new_sum_idx][new_len_parity] = True
        
        dp = new_dp
        reached_non_empty = new_rne
            
    # --- Collect result ---
    final_ans = -1
    target_sum_idx = k + SUM_OFFSET

    # Check if target_sum_idx is valid for the dp table (it should be, due to upfront abs(k) check)
    if not (0 <= target_sum_idx < SUM_RANGE_SIZE):
       return -1 

    # Case 1: Subsequence has target sum k and even length
    val_even_len = dp[target_sum_idx][0]
    is_non_empty_even_len = reached_non_empty[target_sum_idx][0]

    if val_even_len != -1: # If state is reachable
        if is_non_empty_even_len: # And it's confirmed from a non-empty subsequence
            final_ans = max(final_ans, val_even_len)
        # If !is_non_empty_even_len:
        # This implies val_even_len=1, target_sum_idx=SUM_OFFSET (k=0),
        # and this state is the original empty subsequence. We ignore it.

    # Case 2: Subsequence has target sum k and odd length
    val_odd_len = dp[target_sum_idx][1]
    # is_non_empty_odd_len = reached_non_empty[target_sum_idx][1] # Variable not strictly needed

    if val_odd_len != -1: # If state is reachable
        # An odd length subsequence, if its state is valid (val_odd_len != -1),
        # must have been formed by picking at least one element.
        # Thus, reached_non_empty[target_sum_idx][1] must be True.
        # assert reached_non_empty[target_sum_idx][1]
        final_ans = max(final_ans, val_odd_len)
            
    return final_ans