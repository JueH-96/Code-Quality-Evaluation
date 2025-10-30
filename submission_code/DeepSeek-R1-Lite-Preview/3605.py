from typing import List

class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        def find_min_x(num: int) -> int:
            for x in range(num):
                if (x | (x + 1)) == num:
                    return x
            return -1
        
        ans = []
        for num in nums:
            min_x = find_min_x(num)
            ans.append(min_x)
        return ans