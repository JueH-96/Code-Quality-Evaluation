from typing import List

class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n = len(nums)
        m = len(andValues)
        INF = 10**18
        
        # dp[i][j]: min sum for first j elements using i segments
        # j from 0..n, i from 0..m
        dp = [[INF] * (n+1) for _ in range(m+1)]
        dp[0][0] = 0
        
        for i in range(1, m+1):
            # build segment tree on dp[i-1][0..n]
            dp_prev = dp[i-1]
            size = n+1
            # next power of 2
            N = 1
            while N < size:
                N <<= 1
            st = [INF] * (2 * N)
            # fill leaves
            for idx in range(size):
                st[N + idx] = dp_prev[idx]
            for idx in range(N-1, 0, -1):
                st[idx] = min(st[2*idx], st[2*idx+1])
            
            def range_min(l: int, r: int) -> int:
                # inclusive l..r
                res = INF
                l += N
                r += N
                while l <= r:
                    if (l & 1) == 1:
                        res = min(res, st[l])
                        l += 1
                    if (r & 1) == 0:
                        res = min(res, st[r])
                        r -= 1
                    l //= 2
                    r //= 2
                return res
            
            target = andValues[i-1]
            # prevList for AND subarrays ending at j-1
            prevList = []
            
            # compute dp[i][j] for j=1..n
            for j in range(1, n+1):
                # build new AND list for subarrays ending at j-1
                newList = [(nums[j-1], j-1)]
                for val, start in prevList:
                    new_val = val & nums[j-1]
                    # merge if same as last
                    if newList[-1][0] == new_val:
                        # update earliest start
                        if start < newList[-1][1]:
                            newList[-1] = (new_val, start)
                    else:
                        newList.append((new_val, start))
                prevList = newList
                
                # find if target AND exists, and its interval of starts
                # prevList is in order of descending start
                L = -1
                R = -1
                # scan to find the position p
                for p, (val, start_p) in enumerate(prevList):
                    if val == target:
                        L = start_p
                        if p > 0:
                            # previous entry has larger start
                            R = prevList[p-1][1] - 1
                        else:
                            # use full segment up to j-1
                            R = j-1
                        break
                if L != -1 and L <= R:
                    best_prev = range_min(L, R)
                    if best_prev < INF:
                        dp[i][j] = best_prev + nums[j-1]
                # else dp[i][j] stays INF
            
            # after computing dp[i], continue to next segment
        
        ans = dp[m][n]
        return ans if ans < INF else -1