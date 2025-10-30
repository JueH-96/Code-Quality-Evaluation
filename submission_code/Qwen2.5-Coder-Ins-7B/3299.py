class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        from collections import Counter
        count = Counter(nums)
        max_length = 1
        
        for num in count:
            length = 1
            current = num
            while count[current * current] > 0:
                current *= current
                length += 2
            if length % 2 == 0:
                length -= 1
            max_length = max(max_length, length)
        
        return max_length