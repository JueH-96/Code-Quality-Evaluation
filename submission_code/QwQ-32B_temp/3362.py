from collections import defaultdict
from typing import List

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        D = len(set(nums))
        T = n * (n + 1) // 2
        K = (T - 1) // 2
        
        low = 1
        high = D
        
        def count_at_most(m):
            freq = defaultdict(int)
            left = 0
            distinct = 0
            res = 0
            for right in range(n):
                x = nums[right]
                if freq[x] == 0:
                    distinct += 1
                freq[x] += 1
                while distinct > m:
                    y = nums[left]
                    freq[y] -= 1
                    if freq[y] == 0:
                        distinct -= 1
                    left += 1
                res += (right - left + 1)
            return res
        
        while low < high:
            mid = (low + high) // 2
            cnt = count_at_most(mid)
            if cnt >= K + 1:
                high = mid
            else:
                low = mid + 1
        return low