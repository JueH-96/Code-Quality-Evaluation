from typing import List

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        max_num = 0
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    if len({i, j, k}) == 3:
                        binary_str = f"{nums[i]:b}{nums[j]:b}{nums[k]:b}"
                        max_num = max(max_num, int(binary_str, 2))
        return max_num