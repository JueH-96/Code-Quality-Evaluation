from typing import List

class Solution:
    def minCost(self, nums: List[int]) -> int:
        n = len(nums)
        # We use dp[i][k], where:
        #  i = next index in nums to consider for "new" elements (0..n)
        #  k = pending index + 1 (so k=0 means no pending element, k>0 means pending= k-1)
        # Valid states have k <= i.
        # dp[i][k] = minimum cost to remove all elements from the current array state,
        # where we've consumed original indices < i (except the pending, if any), and
        # there may be one pending element at index p = k-1.
        INF = 10**30
        # dp[i] has length i+1 (for k = 0..i)
        dp = [ [INF] * (i+1) for i in range(n+1) ]
        
        # Fill dp bottom-up in decreasing i
        for i in range(n, -1, -1):
            # k runs from 0..i (k=0 => no pending, k>0 => pending at p=k-1)
            for k in range(i+1):
                p = k - 1
                rem_new = n - i
                total_rem = rem_new + (1 if p >= 0 else 0)
                
                # If fewer than 3 elements remain in the current array, we remove them all at once
                if total_rem < 3:
                    if total_rem == 0:
                        dp[i][k] = 0
                    else:
                        # cost is max over the pending (if any) and all new elements i..n-1
                        m = 0
                        if p >= 0:
                            m = nums[p]
                        for j in range(i, n):
                            if nums[j] > m:
                                m = nums[j]
                        dp[i][k] = m
                else:
                    # We have at least 3 elements in the current array.
                    # The front three are:
                    #   if no pending (p<0):   nums[i], nums[i+1], nums[i+2]
                    #   if pending    (p>=0):  nums[p], nums[i], nums[i+1]
                    if p < 0:
                        # No pending
                        a, b, c = i, i+1, i+2
                        ni = i + 3
                        # Try removing each pair among (a,b,c)
                        # remove (a,b)
                        cost = nums[a] if nums[a] > nums[b] else nums[b]
                        best = cost + dp[ni][c+1]
                        # remove (a,c)
                        cost = nums[a] if nums[a] > nums[c] else nums[c]
                        v = cost + dp[ni][b+1]
                        if v < best: best = v
                        # remove (b,c)
                        cost = nums[b] if nums[b] > nums[c] else nums[c]
                        v = cost + dp[ni][a+1]
                        if v < best: best = v
                        dp[i][k] = best
                    else:
                        # One pending at index p
                        a, b, c = p, i, i+1
                        ni = i + 2
                        # remove (a,b)
                        cost = nums[a] if nums[a] > nums[b] else nums[b]
                        best = cost + dp[ni][c+1]
                        # remove (a,c)
                        cost = nums[a] if nums[a] > nums[c] else nums[c]
                        v = cost + dp[ni][b+1]
                        if v < best: best = v
                        # remove (b,c)
                        cost = nums[b] if nums[b] > nums[c] else nums[c]
                        v = cost + dp[ni][a+1]
                        if v < best: best = v
                        dp[i][k] = best
        
        # Start with i=0 and no pending (k=0)
        return dp[0][0]