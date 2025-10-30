from collections import Counter

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        count = Counter(nums)
        max_length = 1
        
        for num in count:
            if num == 1:
                max_length = max(max_length, (count[num] // 3) * 2 + 1)
                continue
            current_length = 2
            while num * num in count:
                num *= num
                current_length += 2
            max_length = max(max_length, current_length - (current_length % 2))
        
        return max_length