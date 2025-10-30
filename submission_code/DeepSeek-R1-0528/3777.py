class Solution:
    def maxProduct(self, nums: List[int], k: int, limit: int) -> int:
        dp = {}
        over = set()
        
        for num in nums:
            new_dp = dict(dp)
            new_over = set(over)
            
            key_start = (num, 1)
            if num <= limit:
                if key_start in new_dp:
                    if num > new_dp[key_start]:
                        new_dp[key_start] = num
                else:
                    new_dp[key_start] = num
                if key_start in new_over:
                    new_over.remove(key_start)
            else:
                if key_start not in new_dp:
                    new_over.add(key_start)
            
            for (alt_prev, par_prev), prod_prev in dp.items():
                sign = 1 if par_prev == 0 else -1
                new_alt = alt_prev + sign * num
                new_par = 1 - par_prev
                key = (new_alt, new_par)
                new_prod = prod_prev * num
                if new_prod <= limit:
                    if key in new_dp:
                        if new_prod > new_dp[key]:
                            new_dp[key] = new_prod
                    else:
                        new_dp[key] = new_prod
                    if key in new_over:
                        new_over.remove(key)
                else:
                    if key not in new_dp:
                        new_over.add(key)
            
            for (alt_prev, par_prev) in over:
                sign = 1 if par_prev == 0 else -1
                new_alt = alt_prev + sign * num
                new_par = 1 - par_prev
                key = (new_alt, new_par)
                if num == 0:
                    if key not in new_dp:
                        new_dp[key] = 0
                        if key in new_over:
                            new_over.remove(key)
                else:
                    if key not in new_dp:
                        new_over.add(key)
            
            dp = new_dp
            over = new_over
        
        ans = -1
        for (alt, par), prod in dp.items():
            if alt == k:
                if prod > ans:
                    ans = prod
        return ans if ans != -1 else -1