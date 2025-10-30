from typing import List

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        s = n * (n + 1) // 2
        if s == 0:
            return 0  # or handle as appropriate, though nums length is at least 1
        
        # Compute the target position for the median
        target = (s - 1) // 2 + 1
        
        # Find the number of distinct elements in nums (k)
        k = len(set(nums))
        
        # Function to compute number of subarrays with at most m distinct elements
        def at_most_m(m):
            count = 0
            left = 0
            freq = {}
            for right in range(n):
                if nums[right] not in freq:
                    freq[nums[right]] = 0
                freq[nums[right]] += 1
                while len(freq) > m:
                    freq[nums[left]] -= 1
                    if freq[nums[left]] == 0:
                        del freq[nums[left]]
                    left += 1
                count += right - left + 1
            return count
        
        # Binary search for the median
        low = 1
        high = k
        while low < high:
            mid = (low + high) // 2
            if at_most_m(mid) >= target:
                high = mid
            else:
                low = mid + 1
        return low