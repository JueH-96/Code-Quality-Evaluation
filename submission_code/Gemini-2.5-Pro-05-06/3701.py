import math

class Solution:
  def minCostGoodCaption(self, caption: str) -> str:
    n = len(caption)
    if n == 0:
      return ""

    ord_a = ord('a')
    caption_ords = [ord(c) - ord_a for c in caption]

    # dp[i] stores (cost, prev_idx, char_code_for_last_block)
    # for the prefix of length i.
    dp = {0: (0, -1, -1)} # Base case: empty prefix

    # Memoization for block costs and comparisons
    memoized_block_costs = {}
    memo_compare = {}

    def get_block_cost(j_start, k_len, char_code_val):
        state = (j_start, k_len, char_code_val)
        if state in memoized_block_costs:
            return memoized_block_costs[state]
        
        cost = 0
        for p_idx in range(j_start, j_start + k_len):
            cost += abs(caption_ords[p_idx] - char_code_val)
        
        memoized_block_costs[state] = cost
        return cost

    def is_smaller(target_len, p1, c1_val, p2, c2_val):
        # Avoid comparing identical options that might arise if an update rule is loose
        if p1 == p2 and c1_val == c2_val:
             return False # Not strictly smaller

        # Memoization for comparison results
        # Sort (p1,c1) and (p2,c2) to make state canonical if comparison is symmetric
        # Here, order matters: is (p1,c1) smaller than (p2,c2)?
        state = (target_len, p1, c1_val, p2, c2_val)
        if state in memo_compare:
            return memo_compare[state]

        s1_blocks = []
        curr_len, curr_p, curr_c = target_len, p1, c1_val
        while curr_len > 0:
            block_k = curr_len - curr_p
            s1_blocks.append((curr_c, block_k))
            if curr_p == 0: break
            # Details from dp table are needed to backtrack previous block's char
            _dp_cost_ignored, dp_prev_p, dp_prev_c = dp[curr_p]
            curr_len, curr_p, curr_c = curr_p, dp_prev_p, dp_prev_c
        s1_blocks.reverse()

        s2_blocks = []
        curr_len, curr_p, curr_c = target_len, p2, c2_val
        while curr_len > 0:
            block_k = curr_len - curr_p
            s2_blocks.append((curr_c, block_k))
            if curr_p == 0: break
            _dp_cost_ignored, dp_prev_p, dp_prev_c = dp[curr_p]
            curr_len, curr_p, curr_c = curr_p, dp_prev_p, dp_prev_c
        s2_blocks.reverse()
        
        idx1, idx2 = 0, 0
        rem1, rem2 = 0, 0 
        
        is_s1_smaller = False
        made_decision = False

        while idx1 < len(s1_blocks) and idx2 < len(s2_blocks):
            if rem1 == 0: rem1 = s1_blocks[idx1][1]
            if rem2 == 0: rem2 = s2_blocks[idx2][1]

            char_s1 = s1_blocks[idx1][0]
            char_s2 = s2_blocks[idx2][0]

            if char_s1 < char_s2:
                is_s1_smaller = True
                made_decision = True
                break 
            if char_s1 > char_s2:
                is_s1_smaller = False
                made_decision = True
                break
            
            consume = min(rem1, rem2)
            rem1 -= consume
            rem2 -= consume
            
            if rem1 == 0: idx1 += 1
            if rem2 == 0: idx2 += 1
        
        if not made_decision:
            # This implies one string is a prefix of another or they are identical.
            # Since both strings for dp[target_len] have length target_len,
            # if no difference is found, they must be identical.
            # By problem spec, multiple good captions could exist.
            # is_smaller should return False if they are identical or s1 is not smaller.
            is_s1_smaller = False

        memo_compare[state] = is_s1_smaller
        return is_s1_smaller

    for i in range(1, n + 1):
        # Initialize dp[i] to a state representing infinity cost
        current_best_for_i = (math.inf, -1, -1) 
        
        for k_len_last_block in range(3, i + 1):
            j_prev_len = i - k_len_last_block
            
            if j_prev_len not in dp: continue # Previous state unreachable
            
            prev_dp_entry = dp[j_prev_len]
            if prev_dp_entry[0] == math.inf: continue # Previous state has infinite cost

            prev_cost = prev_dp_entry[0]
            
            for char_code_val in range(26):
                block_cost = get_block_cost(j_prev_len, k_len_last_block, char_code_val)
                total_cost = prev_cost + block_cost
                
                if total_cost < current_best_for_i[0]:
                    current_best_for_i = (total_cost, j_prev_len, char_code_val)
                elif total_cost == current_best_for_i[0]:
                    # Tie in cost, use lexicographical comparison
                    # current_best_for_i details: (_cost, prev_j_curr, char_val_curr)
                    if is_smaller(i, j_prev_len, char_code_val, 
                                     current_best_for_i[1], current_best_for_i[2]):
                        current_best_for_i = (total_cost, j_prev_len, char_code_val)
        dp[i] = current_best_for_i
    
    final_cost, _, _ = dp.get(n, (math.inf, -1, -1))

    if final_cost == math.inf:
      return ""

    ans_chars_list = []
    curr_reconstruct_len = n
    while curr_reconstruct_len > 0:
        _cost, prev_j_reconstruct, char_val_int = dp[curr_reconstruct_len]
        block_k_reconstruct = curr_reconstruct_len - prev_j_reconstruct
        actual_char = chr(ord_a + char_val_int)
        ans_chars_list.append(actual_char * block_k_reconstruct)
        curr_reconstruct_len = prev_j_reconstruct
        
    return "".join(reversed(ans_chars_list))