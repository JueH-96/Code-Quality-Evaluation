from typing import List
from array import array

class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        half = n // 2

        # Precompute mismatches for fixed positions and their prefix sums
        bad = [0] * half
        for i in range(half):
            if s[i] != s[n - 1 - i]:
                bad[i] = 1
        prefixBad = [0] * (half + 1)
        for i in range(half):
            prefixBad[i + 1] = prefixBad[i] + bad[i]
        totalBad = prefixBad[half]

        # Precompute prefix‐counts for each character over the whole string
        # Use array('I') for memory efficiency
        snums = [ord(ch) - 97 for ch in s]
        prefix = [array('I', [0] * (n + 1)) for _ in range(26)]
        for j in range(26):
            cnt = 0
            arr = prefix[j]
            for i, c in enumerate(snums):
                if c == j:
                    cnt += 1
                arr[i + 1] = cnt

        ans: List[bool] = []
        append = ans.append
        pb = prefixBad
        pref = prefix

        for a, b, c, d in queries:
            # 1) Check mismatches among fixed‐only pairs
            L1, R1 = a, b
            L2, R2 = n - 1 - d, n - 1 - c
            # sum of bad in [L1..R1]
            rem1 = pb[R1 + 1] - pb[L1]
            # sum of bad in [L2..R2]
            rem2 = pb[R2 + 1] - pb[L2]
            # overlap of those two intervals
            iL = L1 if L1 > L2 else L2
            iR = R1 if R1 < R2 else R2
            if iL <= iR:
                rem12 = pb[iR + 1] - pb[iL]
            else:
                rem12 = 0
            fixedBad = totalBad - (rem1 + rem2 - rem12)
            if fixedBad > 0:
                append(False)
                continue

            # Precompute interval endpoints for frequency checks
            a0 = a
            b1 = b + 1
            c0 = c
            d1 = d + 1

            # J_L = mirror of [a..b] => [n-1-b .. n-1-a]
            lJL = n - 1 - b
            rJL = n - 1 - a
            rJL1 = rJL + 1
            # intersection of J_L with right‐segment [c..d]
            lJLint = lJL if lJL > c else c
            rJLint = rJL if rJL < d else d
            haveJLint = (lJLint <= rJLint)

            # J_R = mirror of [c..d] => [n-1-d .. n-1-c]
            lJR = n - 1 - d
            rJR = n - 1 - c
            rJR1 = rJR + 1
            # intersection of J_R with left‐segment [a..b]
            lJRint = lJR if lJR > a else a
            rJRint = rJR if rJR < b else b
            haveJRint = (lJRint <= rJRint)

            # 2) For each character, check if the remaining multisets match
            good = True
            for arr in pref:
                # freq in left‐segment [a..b]
                fL = arr[b1] - arr[a0]
                # freq of mirrors J_L
                fJL = arr[rJL1] - arr[lJL]
                if haveJLint:
                    fJLint = arr[rJLint + 1] - arr[lJLint]
                else:
                    fJLint = 0
                # leftover from left after satisfying single matches
                remL = fL + fJLint - fJL

                # freq in right‐segment [c..d]
                fR = arr[d1] - arr[c0]
                # freq of mirrors J_R
                fJR = arr[rJR1] - arr[lJR]
                if haveJRint:
                    fJRint = arr[rJRint + 1] - arr[lJRint]
                else:
                    fJRint = 0
                # leftover from right after single matches
                remR = fR + fJRint - fJR

                # must be nonnegative and equal on both sides
                if remL < 0 or remL != remR:
                    good = False
                    break

            append(good)

        return ans