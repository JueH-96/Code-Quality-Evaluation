from typing import List
from collections import Counter

class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        n = len(nums)
        MOD = 10**9 + 7
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        for m in range(l + 1, n):
                            subsequence = [nums[i], nums[j], nums[k], nums[l], nums[m]]
                            
                            counts = Counter(subsequence)
                            max_count = 0
                            modes = []
                            for num, freq in counts.items():
                                if freq > max_count:
                                    max_count = freq
                                    modes = [num]
                                elif freq == max_count:
                                    modes.append(num)
                            
                            if len(modes) == 1 and modes[0] == subsequence[2]:
                                count = (count + 1) % MOD
        return count