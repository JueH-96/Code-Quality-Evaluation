class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        from collections import Counter
        
        # Count the frequency of each number in the array
        count = Counter(nums)
        
        # Find the maximum frequency of any number
        max_freq = max(count.values())
        
        # The minimum seconds needed is the total number of elements minus the maximum frequency
        return len(nums) - max_freq