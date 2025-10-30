class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        # Find the minimum value in the array
        min_val = min(nums)
        
        # Check if any element is not divisible by min_val
        for num in nums:
            if num % min_val != 0:
                return 1
        
        # If all elements are divisible by min_val, count occurrences of min_val
        count_min = nums.count(min_val)
        
        # The answer is ceil(count_min / 2)
        return (count_min + 1) // 2