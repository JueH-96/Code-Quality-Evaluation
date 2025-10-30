from typing import List

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        total_subarrays = n * (n + 1) // 2
        
        # The median of a sorted array of length L is the element at index (L-1)//2.
        # This corresponds to the k-th smallest element (1-indexed), where k = (L+1)//2.
        median_k = (total_subarrays + 1) // 2

        def count_less_equal(k: int) -> int:
            """
            Counts the number of subarrays with at most k distinct elements
            using a sliding window approach.
            """
            count = 0
            left = 0
            freq = {}  # Using a standard dictionary for frequency counting
            
            for right in range(n):
                freq[nums[right]] = freq.get(nums[right], 0) + 1
                
                # If the number of distinct elements in the window exceeds k,
                # shrink the window from the left until the condition is met again.
                while len(freq) > k:
                    freq[nums[left]] -= 1
                    if freq[nums[left]] == 0:
                        del freq[nums[left]]
                    left += 1
                
                # At this point, the window nums[left..right] has at most k distinct elements.
                # All subarrays ending at `right` and starting from an index `i`
                # where `left <= i <= right` are valid.
                # The number of such subarrays is (right - left + 1).
                count += (right - left + 1)
            
            return count

        # Binary search for the smallest value `m` (the median) such that
        # the number of subarrays with at most `m` distinct elements is at least `median_k`.
        low, high = 1, n
        ans = 1 # The answer is at least 1.

        while low <= high:
            mid = low + (high - low) // 2
            
            if count_less_equal(mid) >= median_k:
                # `mid` could be the median. Try for a smaller value.
                ans = mid
                high = mid - 1
            else:
                # `mid` is too small, the median must be larger.
                low = mid + 1
    
        return ans