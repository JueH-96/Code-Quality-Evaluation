from typing import List
from collections import defaultdict

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        # Count how many subarrays have at most k distinct elements
        # using a classic sliding-window two-pointer approach.
        # This function runs in O(n) time.
        def count_at_most(k: int) -> int:
            count = 0
            left = 0
            freq = defaultdict(int)
            distinct_count = 0
            
            for right in range(n):
                freq[nums[right]] += 1
                if freq[nums[right]] == 1:
                    distinct_count += 1
                
                # Shrink the window until we have at most k distinct
                while distinct_count > k:
                    freq[nums[left]] -= 1
                    if freq[nums[left]] == 0:
                        distinct_count -= 1
                    left += 1
                
                # Number of subarrays ending at 'right' with at most k distinct
                count += (right - left + 1)
            
            return count
        
        n = len(nums)
        # Total number of subarrays
        total_subarrays = n * (n + 1) // 2
        
        # We want the median index in 1-based indexing:
        # If total_subarrays = L, median index (in 1-based) is (L - 1)//2 + 1,
        # which is equivalently (L + 1)//2.
        K = (total_subarrays + 1) // 2
        
        # The distinct count in any subarray can't exceed
        # the total number of distinct elements in the whole array.
        max_distinct = len(set(nums))
        
        # We'll binary-search the answer in the range [1..max_distinct].
        lo, hi = 1, max_distinct
        
        while lo < hi:
            mid = (lo + hi) // 2
            if count_at_most(mid) >= K:
                hi = mid
            else:
                lo = mid + 1
        
        return lo