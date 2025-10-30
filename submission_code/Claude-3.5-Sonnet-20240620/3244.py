class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        min_num = min(nums)
        count_min = nums.count(min_num)
        
        for num in nums:
            if num % min_num != 0 and num != min_num:
                return 1
        
        return (count_min + 1) // 2