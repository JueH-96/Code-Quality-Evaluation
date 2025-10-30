from collections import defaultdict

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        count = defaultdict(int)
        for num in nums:
            count[num] += 1
        
        max_len = 1
        
        for num in sorted(count):
            if num == 1:
                max_len = max(max_len, count[num] if count[num] % 2 == 1 else count[num] - 1)
                continue
            current_len = 0
            x = num
            while x in count and count[x] >= 2:
                current_len += 2
                x = x * x
            if x in count:
                current_len += 1
            else:
                current_len -= 1
            max_len = max(max_len, current_len)
        
        return max_len