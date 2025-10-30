class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        max_freq = 1
        current_count = 1
        
        for i in range(1, n):
            if nums[i] == nums[i-1]:
                current_count += 1
                if current_count > max_freq:
                    max_freq = current_count
            else:
                current_count = 1
        
        if max_freq > n // 2:
            return 2 * max_freq - n
        else:
            return n % 2