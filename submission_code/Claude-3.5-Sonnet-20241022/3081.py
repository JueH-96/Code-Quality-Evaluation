class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        mid = (n + 1) // 2
        
        # Count frequency of most frequent element
        max_freq = 1
        curr_freq = 1
        for i in range(1, n):
            if nums[i] == nums[i-1]:
                curr_freq += 1
                max_freq = max(max_freq, curr_freq)
            else:
                curr_freq = 1
        
        # If max frequency is greater than half the array length
        # then remaining elements will be max_freq - (n - max_freq)
        if max_freq > n//2:
            return 2 * max_freq - n
            
        # If array length is odd, one element will remain
        return n % 2