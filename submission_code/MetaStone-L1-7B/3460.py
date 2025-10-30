class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        from collections import defaultdict
        
        req = sorted(requirements, key=lambda x: x[1])
        required_ends = [x[1] for x in req]
        required_cnt = [x[0] for x in req]
        required_ends.sort()
        
        m = len(required_ends)
        if m == 0:
            return 1
        
        dp = [defaultdict(int) for _ in range(m)]
        dp[0][0] = 1
        
        for i in range(1, m):
            current_end = required_ends[i]
            prev_end = required_ends[i-1]
            current_cnt = required_cnt[i]
            prev_cnt = required_cnt[i-1]
            
            k = current_end - prev_end
            if k == 0:
                continue
            
            ways = 0
            for prev_inversion in dp[i-1]:
                if prev_inversion != prev_cnt:
                    continue
                
                current_inversion = current_cnt
                if current_inversion < prev_inversion:
                    continue
                
                needed = current_inversion - prev_inversion
                if needed < 0:
                    continue
                
                max_inversion = (k * (k - 1)) // 2
                if needed > max_inversion:
                    continue
                
                for new_inversion in range(needed, max_inversion + 1):
                    if new_inversion == needed:
                        ways += dp[i-1][prev_inversion]
                    else:
                        pass
            
            if needed > max_inversion:
                dp[i][current_inversion] = 0
            else:
                if needed == max_inversion:
                    dp[i][current_inversion] = dp[i-1][prev_inversion]
                else:
                    pass
        
        last_inversion = required_cnt[-1]
        return dp[-1].get(last_inversion, 0)