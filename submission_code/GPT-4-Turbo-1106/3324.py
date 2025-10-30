class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        from collections import Counter
        
        # Count the frequency of each number in nums
        freq = Counter(nums)
        
        # If any number appears more than len(nums) / 2 times, it's impossible to split
        for count in freq.values():
            if count > len(nums) // 2:
                return False
        
        # If we have at least len(nums) / 2 distinct numbers, we can split
        return len(freq) >= len(nums) // 2