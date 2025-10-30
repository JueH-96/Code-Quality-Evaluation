import math

class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)

        # Step 1: Precompute all possible divisors for lengths up to n.
        # divs[l] will contain a list of divisors d of l, where 1 <= d < l.
        # These are potential values for the semi-palindrome property.
        divs = [[] for _ in range(n + 1)]
        for d in range(1, (n // 2) + 1):
            for length in range(2 * d, n + 1, d):
                divs[length].append(d)

        # Step 2: Precompute the minimum changes required for any substring to become a semi-palindrome.
        # costs[i][j] stores this value for the substring s[i...j] (inclusive), i.e., s[i:j+1].
        costs = [[n] * n for _ in range(n)]
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                min_c = n  # A safe upper bound on changes
                
                # Iterate through all valid divisors d for the current substring length.
                for d in divs[length]:
                    current_c = 0
                    # A string is a semi-palindrome for a divisor d if all d subsequences
                    # formed by characters at indices with the same modulo d are palindromes.
                    # We calculate the cost to make each of these d subsequences a palindrome.
                    for rem in range(d):
                        p1 = i + rem
                        p2 = j - (d - 1 - rem)
                        while p1 < p2:
                            if s[p1] != s[p2]:
                                current_c += 1
                            p1 += d
                            p2 -= d
                    min_c = min(min_c, current_c)
                costs[i][j] = min_c

        # Step 3: Use dynamic programming to find the minimum total changes for k partitions.
        # dp[i][j_parts]: minimum cost to partition the prefix s[:i] into j_parts semi-palindromes.
        inf = float('inf')
        dp = [[inf] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0

        # j_parts is the number of partitions.
        for j_parts in range(1, k + 1):
            # i is the length of the prefix s[:i].
            # Minimum length for j_parts partitions is 2 * j_parts, as each needs length >= 2.
            for i in range(2 * j_parts, n + 1):
                # p is the start index of the j_parts-th partition.
                # The first j_parts-1 partitions form s[:p].
                # Min length of s[:p] is 2 * (j_parts-1).
                # The last partition s[p:i] must have length >= 2, so i-p >= 2 => p <= i-2.
                for p in range(2 * (j_parts - 1), i - 1): # p from 2*(j_parts-1) up to i-2
                    if dp[p][j_parts - 1] != inf:
                        # The last partition is the substring s[p...i-1].
                        cost_last_part = costs[p][i - 1]
                        dp[i][j_parts] = min(dp[i][j_parts], dp[p][j_parts - 1] + cost_last_part)
        
        return dp[n][k]