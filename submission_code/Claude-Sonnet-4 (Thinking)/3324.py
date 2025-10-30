class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        from collections import Counter
        
        count = Counter(nums)
        
        for freq in count.values():
            if freq > 2:
                return False
                
        return True