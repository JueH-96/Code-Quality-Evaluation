class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        from collections import Counter
        
        # Count the occurrences of each number
        count = Counter(nums)
        
        # Check if any number appears more than twice
        for num, freq in count.items():
            if freq > 2:
                return False
        
        # If we've made it here, it's possible to split the array
        return True