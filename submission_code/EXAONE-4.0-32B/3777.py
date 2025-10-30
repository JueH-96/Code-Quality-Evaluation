class Solution:
    def maxProduct(self, nums: List[int], k: int, limit: int) -> int:
        max_abs = 1800
        offset = max_abs
        total_states = 2 * max_abs + 1
        
        n = len(nums)
        if n == 0:
            return -1
        
        dp_prev = [[-10**9] * 2 for _ in range(total_states)]
        
        first_num = nums[0]
        if first_num <= limit:
            idx0 = first_num + offset
            if 0 <= idx0 < total_states:
                dp_prev[idx0][1] = first_num
        
        for i in range(1, n):
            dp_curr = [[-10**9] * 2 for _ in range(total_states)]
            num = nums[i]
            for s_val in range(-max_abs, max_abs + 1):
                idx = s_val + offset
                for parity in [0, 1]:
                    prod_val = dp_prev[idx][parity]
                    if prod_val == -10**9:
                        continue
                    if prod_val > dp_curr[idx][parity]:
                        dp_curr[idx][parity] = prod_val
                    
                    if num <= limit:
                        new_idx = num + offset
                        if 0 <= new_idx < total_states:
                            if num > dp_curr[new_idx][1]:
                                dp_curr[new_idx][1] = num
                    
                    sign = 1 if parity == 0 else -1
                    s_append = s_val + sign * num
                    prod_append = prod_val * num
                    new_parity = 1 - parity
                    if s_append < -max_abs or s_append > max_abs:
                        pass
                    elif prod_append > limit:
                        pass
                    else:
                        append_idx = s_append + offset
                        if 0 <= append_idx < total_states:
                            if prod_append > dp_curr[append_idx][new_parity]:
                                dp_curr[append_idx][new_parity] = prod_append
            dp_prev = dp_curr
        
        target_idx = k + offset
        if target_idx < 0 or target_idx >= total_states:
            return -1
        ans = max(dp_prev[target_idx][0], dp_prev[target_idx][1])
        if ans < 0:
            return -1
        return ans