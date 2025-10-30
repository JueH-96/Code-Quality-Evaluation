from typing import List
from collections import defaultdict

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        groups = defaultdict(list)
        for i, num in enumerate(nums):
            groups[num].append(i)
        
        def check(f_length: int) -> bool:
            for pos_list in groups.values():
                if len(pos_list) >= f_length:
                    for i in range(len(pos_list) - f_length + 1):
                        if pos_list[i + f_length - 1] - pos_list[i] <= k + f_length - 1:
                            return True
            return False
        
        left, right = 1, len(nums)
        result = 0
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        return result