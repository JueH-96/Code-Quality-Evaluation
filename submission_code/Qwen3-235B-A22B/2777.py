from typing import List

class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = []
        for i in range(n):
            prefix = nums[:i+1]
            suffix = nums[i+1:]
            pre_distinct = len(set(prefix))
            suf_distinct = len(set(suffix))
            result.append(pre_distinct - suf_distinct)
        return result