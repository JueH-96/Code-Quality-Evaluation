class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        """
        We want to partition s into k contiguous substrings such that the sum of
        changes (character replacements) needed to turn each substring into a
        'semi-palindrome' is minimized.

        A substring t of length L > 1 is called a semi-palindrome if there exists
        an integer d with 1 <= d < L and L % d == 0 such that, for each remainder r
        in [0 .. d-1], the characters at indices r, r+d, r+2d, ... in t form a palindrome
        when read in t. We measure the cost of turning a substring into a semi-palindrome
        by the minimal number of character changes required.

        Approach Outline:
        1) Precompute cost_semipal[i][j]: the cost to change s[i:j+1] into a semi-palindrome.
           - If length of s[i:j+1] = 1, it cannot be a semi-palindrome (cost = very large).
           - Otherwise, find all divisors d of the substring length L (except L itself),
             group characters by index % d, and compute minimal changes to make each group
             a palindrome by majority-character replacement. Take the minimum over all valid d.
        2) Use DP to partition string into k parts. Let dp[x][p] = minimal total cost to 
           partition s[:x] into p substrings. Then:
               dp[x][p] = min(
                   dp[m][p-1] + cost_semipal[m][x-1]
                   for m in range(p-1, x)
               )
        3) The answer is dp[n][k], where n = len(s).

        Complexity:
        - n = len(s) up to 200.
        - Computing cost_semipal for each substring can be done in O(n^2 * L * #divisors(L)).
          Each L <= 200, and #divisors(L) is generally small. With careful implementation,
          this is acceptable.
        - The DP step is O(n^2 * k).

        We return dp[n][k] as the final answer.
        """

        n = len(s)

        # Precompute all divisors for lengths [1..n] once, to save time.
        # divisors[l] = all d in [1..l-1] s.t. l % d == 0
        divisors = [[] for _ in range(n+1)]
        for length in range(2, n+1):
            for d in range(1, length):
                if length % d == 0:
                    divisors[length].append(d)

        # Precompute cost to make s[i:j+1] a semi-palindrome
        INF = 10**9
        cost_semipal = [[INF]*(n) for _ in range(n)]

        # Helper function to compute cost for substring s[i:j+1] if length > 1
        def compute_cost(i, j):
            length = j - i + 1
            if length == 1:
                return INF  # length-1 < 1 => can't be semipal.
            best = INF
            # For every valid divisor d of length
            for d in divisors[length]:
                current_cost = 0
                group_size = length // d
                # For each remainder r in [0..d-1], gather s[i+r], s[i+r+d], ...
                for r in range(d):
                    freq = [0]*26
                    # Collect characters in that group
                    idx = i + r
                    count = 0
                    while idx <= j:
                        freq[ord(s[idx]) - ord('a')] += 1
                        count += 1
                        idx += d
                    # cost to make them all the same is group_size - max_frequency
                    current_cost += (count - max(freq))
                best = min(best, current_cost)
            return best

        # Fill cost_semipal table
        for start in range(n):
            for end in range(start, n):
                cost_semipal[start][end] = compute_cost(start, end)

        # DP array: dp[x][p] = min cost to partition s[:x] into p semi-palindromes
        dp = [[INF]*(k+1) for _ in range(n+1)]
        dp[0][0] = 0  # base case, 0 length into 0 parts => cost 0

        for x in range(1, n+1):
            # p cannot exceed x, and also p <= k
            for p in range(1, min(k, x) + 1):
                # Partition s[:x] into p parts => final block s[m:x]
                # m in [p-1, x-1]
                best_val = INF
                for m in range(p-1, x):
                    # cost of last block is cost_semipal[m][x-1]
                    c = dp[m][p-1] + cost_semipal[m][x-1]
                    if c < best_val:
                        best_val = c
                dp[x][p] = best_val

        return dp[n][k]