class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        """
        We use a binary-lifting (jump) approach to handle potentially large k (up to 1e10).
        - next_table[p][i] will store the node we land on after 2^p passes starting from node i.
        - sum_table[p][i] will store the sum of the node-ids covered in those 2^p passes (not including i itself).
        Then f(x) is computed by adding x once, and repeatedly jumping in powers of two until we've performed k passes.
        The total time complexity is O(n log k), which is feasible for n up to 1e5 and k up to 1e10.
        """
        from typing import List
        from array import array

        n = len(receiver)
        # Number of bits needed to represent k (so 2^maxP > k)
        maxP = k.bit_length()

        # Tables to hold the jump targets and sums at powers of two
        # next_table[p][i]: node reached from i after 2^p passes
        # sum_table[p][i]: sum of node ids visited during those 2^p passes (excluding the starting node)
        next_table = [array('I', [0]*n) for _ in range(maxP)]
        sum_table  = [array('Q', [0]*n) for _ in range(maxP)]

        # Initialize p = 0 (one pass)
        for i in range(n):
            next_table[0][i] = receiver[i]
            sum_table[0][i] = receiver[i]  # sum of exactly one pass (i -> receiver[i])

        # Build the jump tables for p = 1..maxP-1
        for p in range(1, maxP):
            for i in range(n):
                mid = next_table[p - 1][i]
                next_table[p][i] = next_table[p - 1][mid]
                sum_table[p][i] = sum_table[p - 1][i] + sum_table[p - 1][mid]

        ans = 0
        # Compute f(x) for each x using the jump tables
        for x in range(n):
            total_sum = x  # f(x) includes x initially
            current = x
            steps_left = k
            # Decompose steps_left in powers of two
            for p in range(maxP - 1, -1, -1):
                jump_len = 1 << p
                if jump_len <= steps_left:
                    steps_left -= jump_len
                    total_sum += sum_table[p][current]
                    current = next_table[p][current]
            ans = max(ans, total_sum)

        return ans