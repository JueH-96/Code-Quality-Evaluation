from collections import defaultdict

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        freq = defaultdict(int)
        for num in nums:
            freq[num] += 1
        
        max_len = 1
        
        for num in freq:
            if num == 1:
                # For 1, the maximum possible is the count of 1s
                current_len = freq[num]
                if current_len % 2 == 0:
                    current_len -= 1
                max_len = max(max_len, current_len)
                continue
            
            current_len = 1
            x = num
            while True:
                x_squared = x * x
                if x_squared in freq:
                    current_len += 2
                    x = x_squared
                else:
                    break
            # If the current_len is even, we need to subtract 1 to make it odd
            if current_len % 2 == 0:
                current_len -= 1
            max_len = max(max_len, current_len)
        
        return max_len