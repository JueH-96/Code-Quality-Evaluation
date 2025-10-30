class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        """
        We want to partition the string s into k contiguous substrings so that the sum of
        the changes needed to make each substring a 'semi-palindrome' is minimized.

        A string t of length L > 1 is a semi-palindrome if there is a d with 1 <= d < L and L % d == 0
        such that, when grouping the indices of t by i mod d, each group of characters forms a palindrome.

        Approach:
        1) Precompute a 2D array cost[start][end], which gives the minimal number of changes
           needed to transform s[start:end+1] into a semi-palindrome. If the substring length is 1,
           that cost is "infinite" (cannot be semi-pal by definition).

        2) Use a DP approach to find the minimal partition cost:
           dp[j][i] = minimum cost to partition s[:i] (i.e., the first i chars) into j substrings
           that are semi-palindromes. Recurrence:
               dp[j][i] = min( dp[j-1][m] + cost[m][i-1] for m in [j-1..i-1] )
           because we form the j-th substring from s[m..i-1].

        3) The answer is dp[k][n], where n = len(s).

        Time complexity notes:
        - Precomputing cost[start][end] over all substrings can be done in O(n^2 * L * #divisors(L)).
        - Then the DP is O(k * n^2).

        This should be acceptable for n up to 200 with careful implementation.
        """

        n = len(s)
        s_list = list(s)
        INF = 10**9

        # Precompute divisors for lengths up to n
        # We only consider 1 <= d < L if L % d == 0
        divisors = [[] for _ in range(n+1)]
        for length in range(2, n+1):
            for d in range(1, length):
                if length % d == 0:
                    divisors[length].append(d)

        # cost_pal(seq_chars): cost to make seq_chars into a palindrome
        def cost_pal(seq_chars):
            m = len(seq_chars)
            mismatches = 0
            for i in range(m // 2):
                if seq_chars[i] != seq_chars[m - 1 - i]:
                    mismatches += 1
            return mismatches

        # Compute cost to transform s[start:start+length] into a semi-palindrome
        # length >= 2. If length == 1, cost is INF (not semi-pal by definition).
        cost = [[INF] * n for _ in range(n)]

        # Fill cost array for substrings of length >= 2
        for length in range(2, n + 1):
            for start in range(n - length + 1):
                end = start + length - 1
                # Try all valid divisors d of length
                best = INF
                for d in divisors[length]:
                    # We'll sum up the cost of making each "mod-d group" palindrome
                    group_cost_sum = 0
                    group_len = length // d
                    for i_mod in range(d):
                        # We'll check the group s_list[start + i_mod + x*d], x=0..group_len-1
                        # cost_pal on that group
                        left, right = 0, group_len - 1
                        mismatches = 0
                        while left < right:
                            if s_list[start + i_mod + left*d] != s_list[start + i_mod + right*d]:
                                mismatches += 1
                            left += 1
                            right -= 1
                        group_cost_sum += mismatches
                    best = min(best, group_cost_sum)
                cost[start][end] = best

        # Now, DP to partition into k substrings
        # dp[j][i] = min cost to partition s[:i] into j semi-palindromic substrings
        dp = [[INF] * (n + 1) for _ in range(k + 1)]
        dp[0][0] = 0  # 0 substrings for string length 0 => cost 0

        for j in range(1, k + 1):
            # i at least j (but we'll just do up to n)
            for i in range(j, n + 1):
                # We consider the last substring from m to i-1
                # We need at least j-1 characters to form j-1 substrings, so m >= j-1
                for m in range(j - 1, i):
                    c = cost[m][i - 1]
                    curr = dp[j - 1][m] + c
                    if curr < dp[j][i]:
                        dp[j][i] = curr

        return dp[k][n]