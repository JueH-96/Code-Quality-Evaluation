class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        count = {}
        
        # Count the frequency of each number
        for num in nums:
            count[num] = count.get(num, 0) + 1
        
        # Find the maximum frequency
        max_freq = max(count.values())
        
        # If the maximum frequency is greater than half the length of the array
        if max_freq > n // 2:
            return 2 * max_freq - n
        
        # If n is odd, we'll have one element left
        return n % 2