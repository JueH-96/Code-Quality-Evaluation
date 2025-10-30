class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        """
        We want to find max_x ( f(x) ), where:
          f(x) = x + receiver[x] + receiver[receiver[x]] + ... repeated k times

        Observing constraints:
        - n can be up to 1e5
        - k can be up to 1e10

        A direct simulation would be O(k), which is not possible for large k. However, we can
        use a "binary lifting" or "jump table" approach to efficiently compute who we end up
        passing to after some number of steps and also track the sum of the IDs visited.

        Steps:

        1. Build arrays next_node[i][j] and sum_node[i][j] where:
           - next_node[i][0] = receiver[i] (the immediate receiver from i)
           - sum_node[i][0] = receiver[i] (the sum contributed when moving once from i)
           For j > 0:
             next_node[i][j] = next_node[ next_node[i][j-1] ][j-1]
             sum_node[i][j]  = sum_node[i][j-1] + sum_node[ next_node[i][j-1] ][j-1]

        2. For each node x, to compute f(x) = x + (sum of IDs encountered in k passes):
           - Start with current = x, current_sum = x
           - Decompose k in powers of two. For each bit j in k:
             if (k & (1 << j)) is not zero:
                current_sum += sum_node[current][j]
                current = next_node[current][j]
           - This takes O(log k) time per x.

        3. Do this for every x in [0..n-1] and return the maximum f(x).

        Time complexity:
          - Build jump tables: O(n log k)
          - Compute f(x) for all x in [0..n-1]: O(n log k)
          This is acceptable for n=1e5 and log k up to ~34 (k <= 1e10).
        """

        n = len(receiver)

        # 1) Precompute the maximum power of 2 needed for k
        import math
        max_power = max(0, k.bit_length())  # number of bits to represent k

        # 2) Initialize next_node and sum_node
        #   next_node[i][j]: the id we end up at starting from i after 2^j passes
        #   sum_node[i][j]: sum of all node ids visited in those 2^j passes (excluding the starting id)
        next_node = [[0]*max_power for _ in range(n)]
        sum_node = [[0]*max_power for _ in range(n)]

        # Fill level j=0 (1-step jump info):
        # next_node[i][0] = receiver[i]
        # sum_node[i][0]  = receiver[i] (since only that single pass is added)
        for i in range(n):
            nxt = receiver[i]
            next_node[i][0] = nxt
            sum_node[i][0] = nxt

        # Fill the rest using dynamic programming
        for j in range(1, max_power):
            for i in range(n):
                nxt = next_node[i][j-1]
                next_node[i][j] = next_node[nxt][j-1]
                sum_node[i][j] = sum_node[i][j-1] + sum_node[nxt][j-1]

        # 3) Evaluate f(x) for each x
        # f(x) = x + sum_of_ids for k passes
        # We'll use the jump tables to move in O(log k)
        best = 0
        for x in range(n):
            current = x
            current_sum = x  # includes the starting player's id
            remaining = k
            bit = 0
            while remaining > 0:
                if (remaining & 1) == 1:
                    current_sum += sum_node[current][bit]
                    current = next_node[current][bit]
                bit += 1
                remaining >>= 1
            best = max(best, current_sum)

        return best