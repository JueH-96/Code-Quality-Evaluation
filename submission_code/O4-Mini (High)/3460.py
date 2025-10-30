from typing import List

class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10**9 + 7
        
        # Map each required prefix end to its required inversion count
        req = {end: cnt for end, cnt in requirements}
        # We will process requirements in order of increasing prefix end
        ends = sorted(req.keys())
        
        prev_end = -1
        prev_cnt = 0
        answer = 1
        
        for end in ends:
            cnt = req[end]
            # The segment of inversion‐sequence entries we need to fill:
            # indices from L to R inclusive
            L = prev_end + 1
            R = end
            seg_len = R - L + 1
            
            # Required sum on this segment
            k = cnt - prev_cnt
            if k < 0:
                return 0
            
            # Maximum possible sum on this segment is sum of upper‐bounds inv[i] = i
            # for i = L..R
            # sum_i = (L + R) * seg_len // 2
            max_sum = (L + R) * seg_len // 2
            if k > max_sum:
                return 0
            
            # DP array: dp[s] = number of ways to choose inv[L..i-1] summing to s
            dp = [0] * (k + 1)
            dp[0] = 1
            
            # Process each inv[i] with 0 <= inv[i] <= i
            for i in range(L, R + 1):
                ub = i
                # Build prefix sums of dp for fast range‐sum queries
                pref = [0] * (k + 1)
                running = 0
                for s in range(k + 1):
                    running = (running + dp[s]) % MOD
                    pref[s] = running
                
                newdp = [0] * (k + 1)
                # newdp[s] = sum_{t=0..min(ub,s)} dp[s-t]
                for s in range(k + 1):
                    # subtract prefix up to s-ub-1 if in range
                    if s - ub - 1 >= 0:
                        newdp[s] = (pref[s] - pref[s - ub - 1]) % MOD
                    else:
                        newdp[s] = pref[s]
                
                dp = newdp
            
            # dp[k] is the number of ways for this segment
            ways = dp[k]
            answer = (answer * ways) % MOD
            
            # Move to next segment
            prev_end = end
            prev_cnt = cnt
        
        return answer