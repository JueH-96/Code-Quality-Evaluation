class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        len_s = len(s)
        
        def count_up_to(x: int) -> int:
            x_str = str(x)
            len_x = len(x_str)
            if len_x < len_s:
                return 0
            
            prefix_len = len_x - len_s
            
            # Sum for lengths less than len_x
            max_p = len_x - len_s - 1
            if max_p >= 0:
                sum_smaller = (limit + 1) ** max_p
            else:
                sum_smaller = 0
            
            # For exact length len_x
            x_prefix = x_str[:prefix_len]
            x_suffix = x_str[prefix_len:]
            
            # Memoization table for dp
            memo = [[-1, -1] for _ in range(prefix_len + 1)]
            
            def dp(pos: int, tight: int) -> int:
                if pos == prefix_len:
                    return 1
                if memo[pos][tight] != -1:
                    return memo[pos][tight]
                
                ans = 0
                low = 1 if pos == 0 else 0
                dig_upper = int(x_prefix[pos]) if tight else 9
                eff_up = min(limit, dig_upper)
                eff_low = low
                
                if eff_low <= eff_up:
                    for d in range(eff_low, eff_up + 1):
                        new_tight = 1 if tight and d == int(x_prefix[pos]) else 0
                        ans += dp(pos + 1, new_tight)
                
                memo[pos][tight] = ans
                return ans
            
            count_le = dp(0, 1)
            
            # Check if prefix can be equal
            can_equal = True
            for i in range(prefix_len):
                dig = int(x_prefix[i])
                min_dig = 1 if i == 0 else 0
                if not (min_dig <= dig <= limit):
                    can_equal = False
                    break
            
            # Compare suffixes
            s_leq_x_suffix = (s <= x_suffix)
            
            # Calculate number for exact length
            num_exact = count_le - (1 if can_equal and not s_leq_x_suffix else 0)
            
            # Total count
            total = sum_smaller + num_exact
            return total
        
        return count_up_to(finish) - count_up_to(start - 1)