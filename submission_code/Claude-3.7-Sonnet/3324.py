class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        # Count the frequency of each number
        frequency = {}
        for num in nums:
            frequency[num] = frequency.get(num, 0) + 1
        
        # If any number appears more than twice, we can't split
        for count in frequency.values():
            if count > 2:
                return False
        
        return True