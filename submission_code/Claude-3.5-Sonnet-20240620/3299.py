class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        from collections import Counter
        
        count = Counter(nums)
        max_length = 1
        
        # Handle 1 separately
        if 1 in count:
            ones = count[1]
            max_length = max(max_length, ones - (ones % 2 == 0))
            del count[1]
        
        for num in sorted(count.keys()):
            length = 1
            current = num
            while current * current in count and count[current] >= 2:
                length += 2
                count[current] -= 2
                current *= current
            
            if count[current] == 1:
                length += 1
            else:
                length -= 1
            
            max_length = max(max_length, length)
        
        return max_length