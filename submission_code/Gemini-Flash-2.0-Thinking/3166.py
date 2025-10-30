from collections import Counter
from math import ceil, floor

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        n = len(nums)
        counts = Counter(nums)

        for k in range(1, n + 1):
            s = floor(n / k)
            is_valid_k = True
            for count in counts.values():
                if s == 0:
                    if count > 0:
                        is_valid_k = False
                        break
                    continue

                lower_bound = ceil(count / (s + 1))
                upper_bound = floor(count / s)

                if upper_bound < lower_bound:
                    is_valid_k = False
                    break

            if is_valid_k:
                return k

        return n