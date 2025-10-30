class Solution:
    def countNonDecreasingSubarrays(self, nums: List[int], k: int) -> int:
        # n = length of array
        n = len(nums)
        # Build prefix sum array ps so that ps[i] = sum(nums[0:i])
        ps = [0]*(n+1)
        for i in range(n):
            ps[i+1] = ps[i] + nums[i]
        
        # Compute "next greater" array (strictly greater)
        nxt = [n] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] < nums[i]:
                idx = stack.pop()
                nxt[idx] = i
            stack.append(i)
        
        # For every index i, define the cost of “fixing” the block 
        # that starts at i, i.e. if we take the entire block [i, nxt[i]-1],
        # then the cost is:
        #    cost[i] = nums[i] * (nxt[i]-i-1) - (ps[nxt[i]] - ps[i+1])
        cost = [0] * n
        for i in range(n):
            seg_len = nxt[i] - i - 1
            if seg_len > 0:
                cost[i] = nums[i] * seg_len - (ps[nxt[i]] - ps[i+1])
            else:
                cost[i] = 0

        # Build binary lifting tables.
        # For each starting index i let dp[i][0] = nxt[i] and cs[i][0] = cost[i].
        # Then for j >= 1, dp[i][j] will be the index reached after
        # 2^j full blocks with total cost cs[i][j].
        import math
        Lmax = math.floor(math.log2(n)) + 1 if n > 0 else 1
        dp = [[n] * n for _ in range(Lmax)]
        cs = [[0] * n for _ in range(Lmax)]
        for i in range(n):
            dp[0][i] = nxt[i]
            cs[0][i] = cost[i]
        for j in range(1, Lmax):
            for i in range(n):
                nxt_idx = dp[j-1][i]
                if nxt_idx < n:
                    dp[j][i] = dp[j-1][nxt_idx]
                    cs[j][i] = cs[j-1][i] + cs[j-1][nxt_idx]
                else:
                    dp[j][i] = n
                    cs[j][i] = cs[j-1][i]
        
        total = 0
        # For each starting index L, we want the maximum R such that the cost
        # of “fixing” the subarray [L, R] is <= k.
        for L in range(n):
            rem = k  # remaining allowed operations
            cur = L  # current chain pointer (initially, the chain starts at L)
            # Jump as many full blocks as possible (each jump uses cs[][] cost)
            for j in range(Lmax-1, -1, -1):
                nxt_idx = dp[j][cur]
                if nxt_idx < n and cs[j][cur] <= rem:
                    rem -= cs[j][cur]
                    cur = nxt_idx
            # Now our chain is at position "cur".
            # Two cases: either we jumped completely through all blocks,
            # or we are inside a block [cur, nxt[cur]-1] (the current block)
            if cur == n:
                # We have jumped fully and can extend all the way to the end.
                R_max = n - 1
            else:
                block_end = dp[0][cur]  # this is just nxt[cur]
                # In the current block the cost is linear:
                # For any index x in [cur, block_end-1] define:
                #    f(x) = nums[cur]*(x-cur) - (ps[x+1]-ps[cur+1])
                # f(x) is monotonic non-decreasing because for these x we have nums[x+1] <= nums[cur].
                # We use binary search to find the largest index x (in current block) with f(x) <= rem.
                lo, hi = cur, block_end - 1
                best = cur
                while lo <= hi:
                    mid = (lo + hi) // 2
                    cost_here = nums[cur]*(mid - cur) - (ps[mid+1] - ps[cur+1])
                    if cost_here <= rem:
                        best = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                R_max = best
            # Because for fixed L the cost f(L, R) is monotonic in R, every R in [L, R_max] is valid.
            total += (R_max - L + 1)
        return total