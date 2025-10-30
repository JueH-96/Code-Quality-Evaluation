from typing import List

class Solution:
    def maxSum(self, nums: List[int], k: int, m: int) -> int:
        n = len(nums)
        # prefix sums
        pre = [0] * (n + 1)
        for i in range(n):
            pre[i + 1] = pre[i] + nums[i]
        
        # A very small number to represent -infinity
        NEG_INF = -10**18
        
        # dp_selected_prev[j]: max sum using j segments with the j-th segment ending at previous index
        dp_selected_prev = [NEG_INF] * (k + 1)
        # dp_best_prev[j]: max sum using j segments considering up to previous index (whether ending there or not)
        dp_best_prev = [0] + [NEG_INF] * k
        
        # We keep a small history of dp_best arrays so that we can look up dp_best at index (i-m)
        # dp_best_history[0] = dp_best at i-1, dp_best_history[1] = dp_best at i-2, ..., up to length m
        dp_best_history = [dp_best_prev]
        
        for i in range(n):
            dp_not_sel = dp_best_prev  # if we don't select a segment covering i
            
            # Prepare current dp_selected (segment ends at i) and dp_best
            dp_selected_curr = [NEG_INF] * (k + 1)
            
            # Case 1: extend an existing segment
            # Case 2: start a new segment of length exactly m that ends at i (later extensions come via "extend")
            s = i - m + 1
            if s >= 0:
                # sum of the new segment of length m ending at i
                sum_m = pre[i + 1] - pre[s]
                # dp_best at s-1 is stored at dp_best_history[m-1]
                dp_best_s_1 = dp_best_history[m - 1]
                for j in range(1, k + 1):
                    # extend previous segment j
                    v_ext = dp_selected_prev[j] + nums[i]
                    # start a fresh j-th segment of length m
                    v_new = dp_best_s_1[j - 1] + sum_m
                    # take the better of extending or starting new
                    dp_selected_curr[j] = v_ext if v_ext > v_new else v_new
            else:
                # can't start a new segment yet (i < m-1), only extension possible
                for j in range(1, k + 1):
                    dp_selected_curr[j] = dp_selected_prev[j] + nums[i]
            
            # Build dp_best_curr: best of selecting a segment ending at i or not selecting i
            dp_best_curr = [0] * (k + 1)
            for j in range(1, k + 1):
                v_not = dp_not_sel[j]
                v_sel = dp_selected_curr[j]
                dp_best_curr[j] = v_not if v_not > v_sel else v_sel
            # by definition, dp_best_curr[0] = 0 (using 0 segments)
            
            # Update our small history of dp_best arrays
            dp_best_history.insert(0, dp_best_curr)
            if len(dp_best_history) > m:
                dp_best_history.pop()
            
            # Move to next index
            dp_selected_prev = dp_selected_curr
            dp_best_prev = dp_best_curr
        
        # The answer is the best sum using exactly k segments after processing all n elements
        return dp_best_prev[k]