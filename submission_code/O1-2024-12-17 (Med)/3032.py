class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        """
        We want to find the starting player x that maximizes:
            f(x) = x + receiver[x] + receiver[receiver[x]] + ... (k passes total)
        
        We can use a binary-lifting (jump table) approach:
        - jump[l][i]: the player ID reached if we start at i and pass 2^l times.
        - sumJump[l][i]: the sum of the IDs visited in those 2^l passes (excluding the initial i).
        
        Then f(x) = x + (sum of IDs for k passes from x).
        For each x, we combine contributions in a binary decomposition of k.
        """
        import sys
        sys.setrecursionlimit(10**7)
        
        n = len(receiver)
        # Maximum number of bits we might need for k (since k <= 1e10).
        LOG = k.bit_length()  # ~ up to 34 bits for k ~1e10

        # Precompute jump and sumJump
        # jump[l][i] = node we end up at after 2^l passes starting from i
        # sumJump[l][i] = sum of the IDs of the nodes visited in those 2^l passes (not including i)
        jump = [[0]*n for _ in range(LOG)]
        sumJump = [[0]*n for _ in range(LOG)]
        
        # Initialize level 0
        for i in range(n):
            jump[0][i] = receiver[i]
            sumJump[0][i] = receiver[i]
        
        # Build up the tables
        for l in range(1, LOG):
            for i in range(n):
                j = jump[l-1][i]  # next node after 2^(l-1) passes
                jump[l][i] = jump[l-1][j]
                sumJump[l][i] = sumJump[l-1][i] + sumJump[l-1][j]

        def total_sum(start: int, passes: int) -> int:
            """Return the sum of IDs visited starting from 'start' plus 'passes' passes."""
            s = start
            cur = start
            shift = 0
            while passes > 0:
                if (passes & 1) == 1:
                    s += sumJump[shift][cur]
                    cur = jump[shift][cur]
                passes >>= 1
                shift += 1
            return s

        # Compute the maximum f(x) over all x
        ans = 0
        for i in range(n):
            ans = max(ans, total_sum(i, k))
        return ans