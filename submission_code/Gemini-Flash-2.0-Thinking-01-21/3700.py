from typing import List
from collections import Counter

class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        def is_unique_middle_mode(subsequence):
            counts = Counter(subsequence)
            max_freq = 0
            for x in counts:
                max_freq = max(max_freq, counts[x])
            mode_count = 0
            mode_value = None
            for x in counts:
                if counts[x] == max_freq:
                    mode_count += 1
                    mode_value = x
            if mode_count == 1 and mode_value == subsequence[2]:
                return True
            return False

        count = 0
        n = len(nums)
        MOD = 10**9 + 7
        for i1 in range(n - 4):
            for i2 in range(i1 + 1, n - 3):
                for i3 in range(i2 + 1, n - 2):
                    for i4 in range(i3 + 1, n - 1):
                        for i5 in range(i4 + 1, n):
                            subsequence = [nums[i1], nums[i2], nums[i3], nums[i4], nums[i5]]
                            if is_unique_middle_mode(subsequence):
                                count = (count + 1) % MOD
        return count