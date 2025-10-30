class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        """
        We want to split s into k contiguous substrings so that the sum of the minimum changes
        required to convert each substring into a 'semi-palindrome' is minimized.

        Definition of semi-palindrome:
          A string T of length L is a semi-palindrome if there exists a positive integer d
          such that 1 <= d < L, L % d == 0, and if we group characters by their indices modulo d,
          each group forms a palindrome.

        Approach:
         1) Precompute, for every substring s[l:r], the cost to make it into a semi-palindrome.
            - If the length of the substring is 1, it's never a valid semi-palindrome (cost = inf).
            - Otherwise, check all divisors d of the substring length L (with 1 <= d < L).
              For each d:
                a) Form groups for each modulo g in [0..d-1], collect characters s[g], s[g+d], ...
                b) Minimum changes to make one group of length m palindrome is the count of
                   mismatches in the first half vs. the reversed second half.
                c) Sum the changes over all d groups, keep track of the minimum across all d.
         2) Use dynamic programming to find the minimal total cost:
            - Let n = len(s). We define dp[i][j] = minimal cost to partition s[:j] into i substrings.
            - dp[0][0] = 0, and dp[0][j>0] = inf as base conditions.
            - For i from 1..k:
                for j from 1..n:
                  dp[i][j] = min( dp[i-1][l] + cost[l][j] ) over all l < j
            - The answer is dp[k][n].
        """

        n = len(s)
        INF = float('inf')
        s_arr = list(s)  # for faster indexing

        # 1) Precompute cost array: cost[l][r] -> minimum changes to make s[l:r] semi-pal
        #    We'll store in cost[l][r], where r is exclusive.
        cost = [[INF]*(n+1) for _ in range(n+1)]

        # Precompute all divisors for lengths up to n
        # divisors[length] = list of d with 1 <= d < length and length % d == 0
        # we'll also include d=1 if length>1, since 1 is always a divisor
        # (so effectively we get all valid d in [1..L-1] s.t. L%d==0)
        divisors = [[] for _ in range(n+1)]
        for length in range(2, n+1):
            for d in range(1, length):
                if length % d == 0:
                    divisors[length].append(d)

        # Helper to compute minimal changes to make an array of chars a palindrome
        def palindrome_cost(arr):
            c = 0
            left, right = 0, len(arr) - 1
            while left < right:
                if arr[left] != arr[right]:
                    c += 1
                left += 1
                right -= 1
            return c

        for l in range(n):
            for r in range(l+1, n+1):
                L = r - l
                # If the substring length is 1, cost is INF (cannot be semi-pal)
                if L == 1:
                    cost[l][r] = INF
                    continue

                # Build substring array
                sub_arr = s_arr[l:r]
                best = INF
                # Check each divisor d
                for d in divisors[L]:
                    # We want to group indices by modulo g in [0..d-1]
                    # Then sum palindrome cost in each group
                    group_cost_sum = 0
                    group_len = L // d
                    # For each modulo group g
                    for g in range(d):
                        group_chars = []
                        idx = g
                        while idx < L:
                            group_chars.append(sub_arr[idx])
                            idx += d
                        group_cost_sum += palindrome_cost(group_chars)
                    best = min(best, group_cost_sum)
                cost[l][r] = best

        # 2) Dynamic Programming to find minimal total cost with k substrings
        dp = [[INF]*(n+1) for _ in range(k+1)]
        dp[0][0] = 0  # no substrings, no length -> cost 0

        for i in range(1, k+1):
            for j in range(1, n+1):
                # partition s[:j] into i substrings
                best_val = INF
                for mid in range(j):
                    c = cost[mid][j]
                    if c == INF:
                        continue
                    val = dp[i-1][mid] + c
                    if val < best_val:
                        best_val = val
                dp[i][j] = best_val

        return dp[k][n]