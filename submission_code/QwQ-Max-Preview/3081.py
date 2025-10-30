class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        max_count = 1
        current_count = 1
        
        for i in range(1, n):
            if nums[i] == nums[i-1]:
                current_count += 1
                if current_count > max_count:
                    max_count = current_count
            else:
                current_count = 1
        
        option1 = 2 * max_count - n
        option2 = n % 2
        
        return max(option1, option2)