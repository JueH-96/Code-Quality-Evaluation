from typing import List
import collections

class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        D = len(set(nums))
        
        def count_at_most(K):
            n = len(nums)
            left = 0
            count_dist = 0
            freq = collections.Counter()
            res = 0
            for right in range(n):
                if freq[nums[right]] == 0:
                    count_dist += 1
                freq[nums[right]] += 1
                while count_dist > K and left <= right:
                    freq[nums[left]] -= 1
                    if freq[nums[left]] == 0:
                        count_dist -= 1
                    left += 1
                res += (right - left + 1)
            return res
        
        return count_at_most(D) - count_at_most(D - 1)