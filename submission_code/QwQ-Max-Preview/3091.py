class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        from collections import Counter
        cnt = Counter(nums)
        k_zeros = cnt.get(0, 0)
        if 0 in cnt:
            del cnt[0]
        
        max_sum = 0
        dp = [0] * (r + 1)
        dp[0] = 1  # Base case: empty subset
        
        for x, c in cnt.items():
            new_max_sum = min(max_sum + c * x, r)
            new_dp = [0] * (new_max_sum + 1)
            
            for remainder in range(x):
                s_list = list(range(remainder, new_max_sum + 1, x))
                # Compute prefix sums, considering s beyond max_sum as 0
                prefix = [0] * (len(s_list) + 1)
                for i in range(len(s_list)):
                    s = s_list[i]
                    if s <= max_sum:
                        prefix[i+1] = (prefix[i] + dp[s]) % MOD
                    else:
                        prefix[i+1] = prefix[i]
                
                for i in range(len(s_list)):
                    s = s_list[i]
                    if s > new_max_sum:
                        continue
                    m = (s - remainder) // x
                    k_max = min(c, m)
                    start = m - k_max
                    if start < 0:
                        start = 0
                    sum_val = (prefix[m + 1] - prefix[start]) % MOD
                    new_dp[s] = (new_dp[s] + sum_val) % MOD
            
            max_sum = new_max_sum
            # Resize dp to new_max_sum + 1, filling with 0 if necessary
            if len(dp) < len(new_dp):
                dp += [0] * (len(new_dp) - len(dp))
            for i in range(len(new_dp)):
                dp[i] = new_dp[i] % MOD
            # Truncate dp to new_max_sum + 1
            dp = dp[:new_max_sum + 1]
        
        total = 0
        for s in range(l, r + 1):
            if s <= max_sum:
                total = (total + dp[s] * (k_zeros + 1)) % MOD
        return total % MOD