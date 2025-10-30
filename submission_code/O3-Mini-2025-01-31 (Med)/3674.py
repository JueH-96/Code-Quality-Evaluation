from typing import List
import sys
sys.setrecursionlimit(10**6)

class Solution:
    def countNonDecreasingSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Quick check: if the array is already non-decreasing, every subarray (n*(n+1)//2 total) is fixable.
        good = True
        for i in range(1, n):
            if nums[i] < nums[i-1]:
                good = False
                break
        if good:
            return n*(n+1)//2

        # Precompute prefix sums P; P[i] = sum(nums[0..i])
        P = [0] * n
        P[0] = nums[0]
        for i in range(1, n):
            P[i] = P[i-1] + nums[i]

        # Compute next greater-or-equal index (nge) for each index
        nge = [n] * n  # default: n means no later index has a value >= current
        stack = []
        for i in range(n-1, -1, -1):
            # pop until top >= current
            while stack and nums[stack[-1]] < nums[i]:
                stack.pop()
            if stack:
                nge[i] = stack[-1]
            else:
                nge[i] = n
            stack.append(i)

        # For each index i, compute block cost from i up to (nge[i]-1) if we take the entire block.
        # If nge[i] == n, we pretend the block goes from i to n.
        block_cost = [0] * n
        for i in range(n):
            nxt = nge[i]
            # block covers indices i+1 .. (nxt-1)  (if any)
            if nxt - i - 1 > 0:
                # sum for that block is: nums[i] * (nxt-i-1) - (P[nxt-1] - P[i])
                s = P[nxt-1] - P[i] if nxt - 1 >= 0 else 0
                block_cost[i] = nums[i] * (nxt - i - 1) - s
            else:
                block_cost[i] = 0

        # Build sparse table for binary lifting.
        # dp[j][i] = tuple (nxt, cost) where starting at index i, if we jump 2^j blocks,
        # we will reach index dp[j][i].nxt and add dp[j][i].cost.
        # For j == 0: dp[0][i] = (nge[i], block_cost[i])
        import math
        maxJ = (n).bit_length()  # enough levels
        dp = [[(n, 0)] * n for _ in range(maxJ)]
        for i in range(n):
            dp[0][i] = (nge[i], block_cost[i])
        for j in range(1, maxJ):
            for i in range(n):
                nxt_i, cost_i = dp[j-1][i]
                if nxt_i < n:
                    nxt2, cost2 = dp[j-1][nxt_i]
                    dp[j][i] = (nxt2, cost_i + cost2)
                else:
                    dp[j][i] = (n, cost_i)
                    
        # define function to query cost(fix) for subarray starting at index L and ending at R (R>=L)
        # This computes f(L,R) = minimum operations to fix subarray nums[L..R]
        def query_cost(L, R):
            # if subarray length is 1, cost is 0.
            if L == R:
                return 0
            total = 0
            pos = L
            # binary-lift: from high level down, jump as long as the full block lies in [L, R]
            for j in range(maxJ-1, -1, -1):
                nxt, cost_jump = dp[j][pos]
                # dp[j][pos].nxt is the index reached after 2^j blocks.
                if nxt <= R:
                    total += cost_jump
                    pos = nxt
                    if pos >= R:
                        break
            # add the “partial block” from pos to R if needed:
            if pos < R:
                # we are in a block that started at pos; this block would have gone entirely until nge[pos] (if available)
                # but we only take indices pos+1..R.
                # cost for that partial block:
                # = nums[pos]*(R-pos) - (P[R] - P[pos])
                total += nums[pos]*(R - pos) - (P[R] - P[pos])
            return total

        # For each starting index L, use binary search (on R from L to n-1) to find the maximum R (call it max_R)
        # for which f(L,R) <= k.
        # Because f(L,R) is monotonic in R, binary search works.
        res = 0
        for L in range(n):
            lo = L
            hi = n-1
            best = L  # best R we have found (at least the single-element subarray works, cost=0)
            while lo <= hi:
                mid = (lo + hi) // 2
                cost_here = query_cost(L, mid)
                if cost_here <= k:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            res += (best - L + 1)
        return res