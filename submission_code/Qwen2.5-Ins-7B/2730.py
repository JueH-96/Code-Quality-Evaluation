class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        prefix_or = 0
        suffix_or = reduce(or_, nums)
        max_or = 0
        
        for i, num in enumerate(nums):
            new_or = (prefix_or | (num << k) | suffix_or)
            max_or = max(max_or, new_or)
            prefix_or |= num
        
        return max_or