class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        min_num = min(nums)
        
        # Check if any number gives non-zero remainder with min_num
        for num in nums:
            if num % min_num != 0:
                return 1
        
        # Count occurrences of minimum number
        count = sum(1 for x in nums if x == min_num)
        
        # Return ceil(count/2)
        return (count + 1) // 2