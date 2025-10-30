from typing import List

class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        n, m = len(skill), len(mana)

        # prefix times of the 0-th potion on each wizard
        prev = [0] * (n + 1)                 # prev[k] = Σ_{i<k}  time_i0
        for i in range(n):
            prev[i + 1] = prev[i] + skill[i] * mana[0]

        start = 0                            # start time of the current potion
        for j in range(1, m):
            # minimal gap that must be inserted before the j-th potion can start
            gap = prev[1]                    # restriction of wizard 0
            curr = [0] * (n + 1)
            prefix = 0

            for i in range(n):
                prefix += skill[i] * mana[j]     # Σ_{k≤i} time_kj
                curr[i + 1] = prefix

                if i < n - 1:                    # restriction of wizard i+1
                    delta = prev[i + 2] - prefix
                    if delta > gap:
                        gap = delta

            start += gap        # start time of potion j
            prev = curr         # prefix sums of the just-started potion

        # finish time of the last potion on the last wizard
        return start + prev[n]