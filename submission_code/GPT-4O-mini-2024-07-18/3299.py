class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        from collections import Counter
        
        count = Counter(nums)
        max_length = 0
        
        for x in count:
            length = 0
            power = 1
            
            while power <= x:
                if power in count:
                    length += count[power]
                power *= 2
            
            max_length = max(max_length, length)
        
        return max_length