class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        total = 0
        left_freq = defaultdict(int)
        right_freq = defaultdict(int)
        
        # Initialize right_freq with elements from index 1 to n-1
        for i in range(1, n):
            right_freq[nums[i]] += 1
        
        for i in range(n):
            x = nums[i]
            left_x = left_freq.get(x, 0)
            right_x = right_freq.get(x, 0)
            
            sum_left = sum(left_freq.values()) - left_freq.get(x, 0)
            sum_right = sum(right_freq.values()) - right_freq.get(x, 0)
            
            max_a = min(2, left_x)
            max_b = min(2, right_x)
            
            for a in range(0, max_a + 1):
                for b in range(0, max_b + 1):
                    s_left = 2 - a
                    s_right = 2 - b
                    if s_left < 0 or s_right < 0:
                        continue
                    # Check if there are enough elements in left and right parts
                    left_non_x_sum = sum_left
                    right_non_x_sum = sum_right
                    available_left = left_non_x_sum >= s_left
                    available_right = right_non_x_sum >= s_right
                    if not (available_left and available_right):
                        continue
                    
                    K = 1 + a + b
                    valid_ways = 0
                    
                    y_values = set()
                    for y in left_freq:
                        if y != x:
                            y_values.add(y)
                    for y in right_freq:
                        if y != x:
                            y_values.add(y)
                    
                    dp = [[0] * (s_right + 1) for _ in range(s_left + 1)]
                    dp[0][0] = 1
                    
                    for y in y_values:
                        cl = left_freq.get(y, 0)
                        cr = right_freq.get(y, 0)
                        transitions = []
                        
                        max_dl = min(cl, K - 1)
                        for dl in range(0, max_dl + 1):
                            max_dr = min(cr, K - 1 - dl)
                            if max_dr < 0:
                                continue
                            for dr in range(0, max_dr + 1):
                                cnt = math.comb(cl, dl) * math.comb(cr, dr)
                                if cnt > 0:
                                    transitions.append((dl, dr, cnt))
                        
                        new_dp = [[0] * (s_right + 1) for _ in range(s_left + 1)]
                        for cl_sum in range(s_left + 1):
                            for cr_sum in range(s_right + 1):
                                if dp[cl_sum][cr_sum] == 0:
                                    continue
                                for (dl, dr, cnt) in transitions:
                                    new_cl = cl_sum + dl
                                    new_cr = cr_sum + dr
                                    if new_cl <= s_left and new_cr <= s_right:
                                        new_dp[new_cl][new_cr] = (new_dp[new_cl][new_cr] + dp[cl_sum][cr_sum] * cnt) % MOD
                        dp = new_dp
                    
                    valid_ways = dp[s_left][s_right] if s_left <= (left_non_x_sum) and s_right <= (right_non_x_sum) else 0
                    ways_x = math.comb(left_x, a) * math.comb(right_x, b)
                    total = (total + valid_ways * ways_x) % MOD
            
            # Update left_freq and right_freq
            if i + 1 < n:
                right_freq[nums[i + 1]] -= 1
                if right_freq[nums[i + 1]] == 0:
                    del right_freq[nums[i + 1]]
            left_freq[x] += 1
        
        return total % MOD