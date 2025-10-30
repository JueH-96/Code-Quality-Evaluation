from typing import List

class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        # We are to compute f(x) = x + receiver[x] + receiver[receiver[x]] + ... for k passes,
        # i.e. (k+1) terms starting with x.
        # We'll use binary lifting (sparse table) to quickly compute f(x) for each starting x.
        # For binary lifting, define:
        #   dp[p][i] = the node reached after 2^p passes from node i.
        #   sumdp[p][i] = sum of the nodes encountered in those 2^p passes, excluding the starting node.
        # Note: For one pass (p == 0), from node i, we get receiver[i] and add receiver[i].
        #
        # Then for a starting node x, we initialize total = x (for step 0) and then for each set bit in the binary
        # decomposition of k (number of passes), add sumdp accordingly.
        
        max_power = k.bit_length()  # number of bits needed to represent k
        
        # initialize dp and sumdp arrays with dimensions (max_power x n)
        dp = [ [0] * n for _ in range(max_power) ]
        sumdp = [ [0] * n for _ in range(max_power) ]
        
        # Base: for one pass (2^0 passes) from i.
        for i in range(n):
            dp[0][i] = receiver[i]
            sumdp[0][i] = receiver[i]
        
        # Precompute higher levels: for each power p, dp[p][i] gives the node reached after 2^p passes from i
        # and sumdp[p][i] gives sum of nodes encountered in these 2^p passes.
        for p in range(1, max_power):
            for i in range(n):
                nxt = dp[p-1][i]
                dp[p][i] = dp[p-1][nxt]
                sumdp[p][i] = sumdp[p-1][i] + sumdp[p-1][nxt]
        
        # Now compute maximum f(x) for x in [0, n-1]
        # f(x) = x (starting value) plus the sum of nodes reached after exactly k passes.
        best = float('-inf')
        # For each starting player.
        for x in range(n):
            total = x  # include the starting node score.
            cur = x
            rem = k  # k passes remain
            bit = 0
            while rem:
                if rem & 1:
                    total += sumdp[bit][cur]
                    cur = dp[bit][cur]
                rem //= 2
                bit += 1
            if total > best:
                best = total
        return best