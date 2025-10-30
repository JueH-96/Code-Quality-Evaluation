import bisect
from collections import defaultdict

class Solution:
    def maxFrequency(self, nums: list, k: int, numOperations: int) -> int:
        nums.sort()
        freq = defaultdict(int)
        for num in nums:
            freq[num] += 1
        
        candidates = set()
        for num in nums:
            candidates.add(num)
            candidates.add(num + k)
            candidates.add(num - k)
        
        ans = 0
        for x in candidates:
            low_bound = x - k
            high_bound = x + k
            left_index = bisect.bisect_left(nums, low_bound)
            right_index = bisect.bisect_right(nums, high_bound) - 1
            
            if left_index > right_index:
                f = 0
            else:
                f = right_index - left_index + 1
            
            count_x = freq.get(x, 0)
            if f - count_x <= numOperations:
                if f > ans:
                    ans = f
        return ans