from typing import List

class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        # We need to compute for each starting player x:
        #    f(x) = x + receiver[x] + receiver[receiver[x]] + ... after k passes.
        # Because k can be very large (up to 10^10) and n up to 10^5,
        # we cannot simulate the k passes directly.
        # Instead we will use doubling technique (binary lifting) to precompute:
        #   jump[level][x] = node reached when starting at x and making 2^level passes.
        # and
        #   sums[level][x] = sum of node ids visited along these 2^level passes (i.e.,
        #                    f(x) without including x itself).
        #
        # Then for each x, we can compute the total sum f(x) = x + (sum over k passes)
        # by processing the binary representation of k.
        #
        # Initialize level 0:
        max_level = 0
        temp = k
        while temp:
            max_level += 1
            temp //= 2

        # jump[0][i] is simply receiver[i] (1 pass) and sums[0][i] is receiver[i]
        jump = [receiver[:]]
        sums = [receiver[:]]

        # Build the doubling table:
        for level in range(1, max_level):
            next_jump = [0] * n
            next_sum = [0] * n
            prev_jump = jump[level - 1]
            prev_sum = sums[level - 1]
            for i in range(n):
                # After 2^level passes from i, you first do 2^(level-1) passes from i,
                # then another 2^(level-1) passes from that point.
                nxt = prev_jump[i]
                next_jump[i] = prev_jump[nxt]
                next_sum[i] = prev_sum[i] + prev_sum[nxt]
            jump.append(next_jump)
            sums.append(next_sum)

        # For each starting node x, compute f(x) = x + (sum over k passes)
        best = -10**18  # set to very small number initially
        for x in range(n):
            total = x  # include the starting node itself
            cur = x
            temp_k = k
            level = 0
            while temp_k:
                if temp_k & 1:
                    total += sums[level][cur]
                    cur = jump[level][cur]
                temp_k //= 2
                level += 1
            if total > best:
                best = total

        return best

# The following code snippet is for running the solution for test cases.
if __name__ == '__main__':
    # Example testcases:
    sol = Solution()
    # Example 1:
    print(sol.getMaxFunctionValue([2, 0, 1], 4))  # Expected output: 6
    # Example 2:
    print(sol.getMaxFunctionValue([1, 1, 1, 2, 3], 3))  # Expected output: 10