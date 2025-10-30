from typing import List
from collections import defaultdict

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        n = len(nums)
        freq = defaultdict(int)
        count = [0] * (n + 1)
        max_freq = 0
        answer = 0
        left = 0
        
        for right in range(n):
            x = nums[right]
            freq[x] += 1
            count[freq[x]] += 1
            if freq[x] > max_freq:
                max_freq = freq[x]
            
            while (right - left + 1) - max_freq > k:
                y = nums[left]
                freq[y] -= 1
                count[freq[y]] += 1
                if freq[y] == max_freq:
                    count[max_freq] -= 1
                while count[max_freq] == 0 and max_freq > 0:
                    max_freq -= 1
                left += 1
            
            answer = max(answer, max_freq)
        
        return answer