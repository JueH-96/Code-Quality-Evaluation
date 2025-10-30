class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        total_subarrays = n * (n + 1) // 2
        target = (total_subarrays + 1) // 2  # Position of median (1-indexed)
        
        def count_subarrays_with_at_most_k_distinct(k):
            """Count subarrays with at most k distinct elements"""
            count = 0
            left = 0
            freq = {}
            
            for right in range(n):
                # Add current element
                freq[nums[right]] = freq.get(nums[right], 0) + 1
                
                # Shrink window if we have more than k distinct elements
                while len(freq) > k:
                    freq[nums[left]] -= 1
                    if freq[nums[left]] == 0:
                        del freq[nums[left]]
                    left += 1
                
                # All subarrays ending at 'right' and starting from 'left' to 'right'
                # have at most k distinct elements
                count += right - left + 1
            
            return count
        
        # Binary search on the answer
        left, right = 1, len(set(nums))
        
        while left < right:
            mid = (left + right) // 2
            if count_subarrays_with_at_most_k_distinct(mid) >= target:
                right = mid
            else:
                left = mid + 1
        
        return left