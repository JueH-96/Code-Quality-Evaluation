class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        """
        We want to find max f(x) over x in [0..n-1], where:
          f(x) = x + receiver[x] + receiver[receiver[x]] + ... (k passes)

        Observing that k can be as large as 1e10, we must avoid naive simulation.
        Instead, we use a binary-lifting approach:
          - jump[i][p] = the node reached from i by applying 'receiver' 2^p times
          - sum_jump[i][p] = the sum of IDs seen in those 2^p passes (i.e., if we start at i
            and follow 'receiver' 2^p times, we gather the sum of all intermediate
            receiver-IDs in that path)

        We'll build these tables up to p ~ floor(log2(k)) (≈ 34 for k ≤ 1e10).
        Then, for each x:
          1) Start with total = x (the starting id).
          2) Set current = x
          3) Decompose k in powers of two, from high to low bit:
             - if (k >> p) & 1 == 1, then:
                 total += sum_jump[current][p]
                 current = jump[current][p]
          4) Keep track of the maximum total seen.

        This approach has O(n log k) preprocessing and O(n log k) to compute
        the result for all x, which is feasible for n up to 1e5 and log k up to ~34.
        """
        import math

        n = len(receiver)
        # Maximum power p we need for binary lifting (for k up to 1e10, about 34-35 bits)
        max_p = k.bit_length()  # This gives us enough bits to cover up to k

        # Precompute jump[i][0] and sum_jump[i][0]
        jump = [[0]*n for _ in range(max_p)]
        sum_jump = [[0]*n for _ in range(max_p)]

        # Base layer (2^0 = 1 jump)
        for i in range(n):
            jump[0][i] = receiver[i]
            sum_jump[0][i] = receiver[i]  # sum of IDs after 1 pass from i is receiver[i]

        # Build up the binary lifting tables
        for p in range(1, max_p):
            for i in range(n):
                # We jump from i to jump[p-1][i], then from that node again p-1 steps
                mid = jump[p-1][i]
                jump[p][i] = jump[p-1][mid]
                sum_jump[p][i] = sum_jump[p-1][i] + sum_jump[p-1][mid]

        # Compute f(x) for each x and track the maximum
        max_value = 0
        for x in range(n):
            total = x
            curr = x
            steps = k
            bit = 0
            while steps > 0:
                # Highest set bit approach or standard while approach
                # We'll do it from low bit upward for clarity
                if steps & 1:
                    total += sum_jump[bit][curr]
                    curr = jump[bit][curr]
                steps >>= 1
                bit += 1
            max_value = max(max_value, total)

        return max_value