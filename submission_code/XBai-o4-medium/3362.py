from collections import defaultdict
from typing import List

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        total_subarrays = n * (n + 1) // 2
        k = (total_subarrays - 1) // 2
        max_unique = len(set(nums))
        
        def count_at_most(K):
            if K == 0:
                return 0
            freq = defaultdict(int)
            left = 0
            res = 0
            for right in range(n):
                num = nums[right]
                freq[num] += 1
                # Shrink the window until the distinct count is <= K
                while len(freq) > K:
                    left_num = nums[left]
                    freq[left_num] -= 1
                    if freq[left_num] == 0:
                        del freq[left_num]
                    left += 1
                # Add the number of valid subarrays ending at 'right'
                res += right - left + 1
            return res
        
        low = 1
        high = max_unique
        ans = max_unique
        while low <= high:
            mid = (low + high) // 2
            cnt = count_at_most(mid)
            if cnt > k:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        return ans