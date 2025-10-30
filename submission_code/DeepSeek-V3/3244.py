class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        min_num = min(nums)
        # Check if any element is not divisible by min_num
        for num in nums:
            if num % min_num != 0:
                return 1
        # Count the number of min_num in the array
        count = nums.count(min_num)
        # The minimum length is ceil(count / 2)
        return (count + 1) // 2