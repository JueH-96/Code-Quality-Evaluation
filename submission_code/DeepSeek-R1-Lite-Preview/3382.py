class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        total = 0
        n = len(nums)
        if n == 0:
            return 0
        
        current_value = nums[0]
        count = 1
        
        for i in range(1, n):
            if nums[i] == current_value:
                count += 1
            else:
                total += count * (count + 1) // 2
                current_value = nums[i]
                count = 1
        
        # Add the last group
        total += count * (count + 1) // 2
        
        return total