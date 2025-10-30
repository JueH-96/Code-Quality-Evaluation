class Solution:
    def punishmentNumber(self, n: int) -> int:
        """
        For every integer i in [1, n] check whether its square can be split into
        contiguous substrings whose integer values sum to i.  If so, add i*i to
        the running total.
        
        Because n ≤ 1000 the square of any i has at most 7 decimal digits
        (1000² = 1 000 000).  A brute-force DFS over all possible split points
        therefore explores at most 2⁷ = 128 states per number – completely
        negligible.
        """

        # helper: can the square of i be partitioned as required?
        def good(i: int) -> bool:
            s = str(i * i)
            length = len(s)

            # depth-first search over split positions
            def dfs(idx: int, accumulated: int) -> bool:
                if idx == length:              # consumed the whole string
                    return accumulated == i    # success if sums to i

                val = 0
                for j in range(idx, length):
                    val = val * 10 + int(s[j])     # extend current substring
                    # pruning: the partial sum already exceeds i → impossible
                    if accumulated + val > i:
                        break
                    if dfs(j + 1, accumulated + val):
                        return True
                return False

            return dfs(0, 0)

        total = 0
        for num in range(1, n + 1):
            if good(num):
                total += num * num
        return total