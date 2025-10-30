class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Find the frequency of the most frequent element
        max_freq = 1
        current_freq = 1
        
        for i in range(1, n):
            if nums[i] == nums[i-1]:
                current_freq += 1
                max_freq = max(max_freq, current_freq)
            else:
                current_freq = 1
        
        return max(n % 2, 2 * max_freq - n)