class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        def compute_max_run(s_str):
            if not s_str:
                return 0
            max_run = 1
            current_run = 1
            for i in range(1, len(s_str)):
                if s_str[i] == s_str[i-1]:
                    current_run += 1
                    if current_run > max_run:
                        max_run = current_run
                else:
                    current_run = 1
            return max_run
        
        original_max = compute_max_run(s)
        if original_max == 1:
            return 1
        
        def is_possible(L):
            s_int = [int(c) for c in s]
            n = len(s_int)
            prev_dp = {}
            c0 = s_int[0]
            prev_dp[(c0, 1)] = 0
            flipped_c0 = 1 - c0
            prev_dp[(flipped_c0, 1)] = 1
            
            for i in range(1, n):
                current_char = s_int[i]
                current_dp = {}
                for (prev_c, prev_len), cost in prev_dp.items():
                    # Option 1: Do not flip
                    new_c = current_char
                    if new_c == prev_c:
                        new_len = prev_len + 1
                        if new_len <= L:
                            key = (new_c, new_len)
                            if key not in current_dp or cost < current_dp.get(key, float('inf')):
                                current_dp[key] = cost
                    else:
                        key = (new_c, 1)
                        if key not in current_dp or cost < current_dp.get(key, float('inf')):
                            current_dp[key] = cost
                    
                    # Option 2: Flip
                    new_c = 1 - current_char
                    new_cost = cost + 1
                    if new_c == prev_c:
                        new_len = prev_len + 1
                        if new_len <= L:
                            key = (new_c, new_len)
                            if key not in current_dp or new_cost < current_dp.get(key, float('inf')):
                                current_dp[key] = new_cost
                    else:
                        key = (new_c, 1)
                        if key not in current_dp or new_cost < current_dp.get(key, float('inf')):
                            current_dp[key] = new_cost
                prev_dp = current_dp
                if not prev_dp:
                    break  # Early termination if no possible states
            
            # Check if any state's cost is within the allowed operations
            for cost in prev_dp.values():
                if cost <= numOps:
                    return True
            return False
        
        low = 1
        high = original_max
        answer = original_max  # Initialize with the highest possible value
        
        while low < high:
            mid = (low + high) // 2
            if is_possible(mid):
                high = mid
                answer = mid
            else:
                low = mid + 1
        
        return answer