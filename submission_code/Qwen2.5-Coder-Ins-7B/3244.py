class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        min_val = min(nums)
        count = nums.count(min_val)
        if count % 2 == 0:
            return count // 2
        else:
            return (count // 2) + 1