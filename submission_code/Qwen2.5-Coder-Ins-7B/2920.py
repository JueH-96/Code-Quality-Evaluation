class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        nums += nums  # Duplicate the array to handle circular nature
        max_time = 0
        
        for num in set(nums):
            time = 0
            last_index = -n
            for i in range(n):
                if nums[i] == num:
                    time = max(time, (i - last_index) // 2)
                    last_index = i
            max_time = max(max_time, time)
        
        return max_time