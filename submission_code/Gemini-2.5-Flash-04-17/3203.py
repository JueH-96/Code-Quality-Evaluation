from typing import List

class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        n_half = n // 2

        # Precompute first and last mismatch indices in the left half [0, n/2 - 1]
        # s[i] vs s[n - 1 - i]
        first_bad = -1
        last_bad = -1
        for i in range(n_half):
            if s[i] != s[n - 1 - i]:
                if first_bad == -1:
                    first_bad = i
                last_bad = i

        # Precompute prefix counts of characters
        # pref_cnt[k][j] = count of char with index j (0-indexed) in s[0...k-1]
        pref_cnt = [[0] * 26 for _ in range(n + 1)]
        for i in range(n):
            for j in range(26):
                pref_cnt[i + 1][j] = pref_cnt[i][j]
            pref_cnt[i + 1][ord(s[i]) - ord('a')] += 1

        def get_count(l, r, char_idx):
            # Count of char_idx in s[l...r] (inclusive)
            if l > r:
                return 0
            return pref_cnt[r + 1][char_idx] - pref_cnt[l][char_idx]

        results = []
        for a, b, c, d in queries:
            # Map right half rearrangement range [c, d] to corresponding range in left half [c_prime, d_prime]
            # An index i in [0, n/2 - 1] corresponds to n - 1 - i in [n/2, n - 1].
            # If n - 1 - i is in [c, d], then c <= n - 1 - i <= d, which means n - 1 - d <= i <= n - 1 - c.
            c_prime = n - 1 - d
            d_prime = n - 1 - c

            # The indices in the left half [0, n/2 - 1] that are potentially affected by rearrangements
            # are those in [a, b] (left rearrangeable) or those corresponding to [c, d] (right rearrangeable), which are in [c_prime, d_prime].
            # The union of these ranges is [a, b] U [c_prime, d_prime].
            # This union covers the range [min(a, c_prime), max(b, d_prime)].
            start_i = min(a, c_prime)
            end_i = max(b, d_prime)

            # Check 1: Fixed parts must match.
            # Any mismatch s[i] != s[n-1-i] for i in [0, n/2 - 1] must occur within the potentially affected range [start_i, end_i].
            # If there is any mismatch (first_bad != -1), the first mismatch index must be >= start_i and the last mismatch index must be <= end_i.
            if first_bad != -1:
                if first_bad < start_i or last_bad > end_i:
                    results.append(False)
                    continue

            # Check 2: Character counts in involved segments must balance parity.
            # The indices involved in forming palindrome pairs (s'[i], s'[n-1-i]) for i in [start_i, end_i]
            # are the characters originally at s[i] for i in [start_i, end_i] and s[n-1-i] for i in [start_i, end_i].
            # The corresponding indices in the right half are [n-1-end_i, n-1-start_i].
            start_j = n - 1 - end_i
            end_j = n - 1 - start_i

            # The multiset of characters from s[start_i : end_i + 1] combined with the multiset from s[start_j : end_j + 1]
            # must contain an even count for each character. This is necessary to form pairs (ch, ch) required for a palindrome
            # across these corresponding segments.
            # This is equivalent to checking if the count of each character in s[start_i : end_i + 1] has the same parity
            # as the count of the same character in s[start_j : end_j + 1].

            possible = True
            for j in range(26): # Iterate through 'a' to 'z'
                count_left_involved = get_count(start_i, end_i, j)
                count_right_involved = get_count(start_j, end_j, j)

                if (count_left_involved % 2) != (count_right_involved % 2):
                    possible = False
                    break

            results.append(possible)

        return results