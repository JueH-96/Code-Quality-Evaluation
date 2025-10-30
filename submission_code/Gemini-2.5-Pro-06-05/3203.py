import collections
from typing import List

class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        n_half = n // 2

        # s1 is the first half, t is the reversed second half
        s1 = s[:n_half]
        t = s[n_half:][::-1]

        # p_mismatch[i] = number of mismatches in s1[0...i-1] vs t[0...i-1]
        p_mismatch = [0] * (n_half + 1)
        for i in range(n_half):
            p_mismatch[i + 1] = p_mismatch[i]
            if s1[i] != t[i]:
                p_mismatch[i + 1] += 1

        # prefix_s1[i][char_code] = count of char in s1[0...i-1]
        prefix_s1 = [[0] * 26 for _ in range(n_half + 1)]
        prefix_t = [[0] * 26 for _ in range(n_half + 1)]
        for i in range(n_half):
            for j in range(26):
                prefix_s1[i + 1][j] = prefix_s1[i][j]
                prefix_t[i + 1][j] = prefix_t[i][j]
            prefix_s1[i + 1][ord(s1[i]) - ord('a')] += 1
            prefix_t[i + 1][ord(t[i]) - ord('a')] += 1

        ans = []
        for a, b, c, d in queries:
            # Map second half indices [c, d] to first half's coordinate system
            # This corresponds to range [a', b'] in t
            a_prime = n - 1 - d
            b_prime = n - 1 - c

            # The union of the two rearrangeable ranges [a, b] and [a', b']
            # defines an effective "mobile" zone.
            L = min(a, a_prime)
            R = max(b, b_prime)

            # 1. Check for mismatches in the parts of the string that are entirely
            # outside the influence of the rearrangements.
            # These parts must already match.
            
            # Mismatches in s1[0...L-1] vs t[0...L-1]
            if p_mismatch[L] > 0:
                ans.append(False)
                continue
            
            # Mismatches in s1[R+1...n_half-1] vs t[R+1...n_half-1]
            if (p_mismatch[n_half] - p_mismatch[R + 1]) > 0:
                ans.append(False)
                continue

            # 2. For the combined "mobile" range [L, R], the multisets of characters
            # in s1[L...R] and t[L...R] must be identical. If they are not anagrams, 
            # it's impossible to make them equal.
            
            s1_counts = [prefix_s1[R + 1][j] - prefix_s1[L][j] for j in range(26)]
            t_counts = [prefix_t[R + 1][j] - prefix_t[L][j] for j in range(26)]

            if s1_counts == t_counts:
                ans.append(True)
            else:
                ans.append(False)
                
        return ans