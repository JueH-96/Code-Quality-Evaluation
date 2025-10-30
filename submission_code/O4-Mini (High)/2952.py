from typing import List

class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        S0 = sum(nums1)
        # If initial sum already <= x, no time needed
        if S0 <= x:
            return 0
        
        # Precompute total increment sum
        S2 = sum(nums2)
        
        # Pair up and sort indices by descending nums2 (b_i)
        # We'll do DP in that order
        pairs = sorted(zip(nums2, nums1), key=lambda p: -p[0])
        b = [p[0] for p in pairs]  # sorted descending
        w = [p[1] for p in pairs]
        
        # Binary search on time T in [1..n]
        left, right = 1, n
        ans = -1
        
        # A large negative for DP initialization
        NEG = -10**18
        
        while left <= right:
            mid = (left + right) // 2
            T = mid
            # Required total reduction to bring sum <= x at time T
            # baseSum = S0 + T*S2, so need reduction D = baseSum - x
            D = S0 + T * S2 - x
            # If no reduction needed
            if D <= 0:
                ans = T
                right = mid - 1
                continue
            
            # We can use at most K = min(n, T) resets
            K = T if T < n else n
            
            # dp[t] = max total reduction using exactly t resets among processed indices
            dp = [0] + [NEG] * K
            
            # DP over sorted indices
            for i in range(n):
                bi = b[i]
                wi = w[i]
                # Precompute part of weight: wi + bi * T
                base = wi + bi * T
                # We can pick at most one reset per index, t goes from min(i+1,K) down to 1
                m = i+1 if i+1 < K else K
                # Update dp backwards to avoid overwrite
                for t in range(m, 0, -1):
                    # If we use this index as the t-th reset (in order of descending b),
                    # the slot time is T - (t-1), so reduction = wi + bi*(T - (t-1))
                    # = base - bi*(t-1)
                    val = dp[t-1] + base - bi * (t-1)
                    if val > dp[t]:
                        dp[t] = val
            
            # After DP, dp[K] is the maximum total reduction using K resets
            if dp[K] >= D:
                ans = T
                right = mid - 1
            else:
                left = mid + 1
        
        return ans