class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Binary search on the answer
        def countSubarraysWithAtMostKDistinct(k):
            count = 0
            left = 0
            freq = {}
            
            for right in range(n):
                if nums[right] in freq:
                    freq[nums[right]] += 1
                else:
                    freq[nums[right]] = 1
                
                while len(freq) > k:
                    freq[nums[left]] -= 1
                    if freq[nums[left]] == 0:
                        del freq[nums[left]]
                    left += 1
                
                count += right - left + 1
            
            return count
        
        # Total number of subarrays
        total = n * (n + 1) // 2
        
        # Binary search for the median
        left, right = 1, n
        
        while left < right:
            mid = (left + right) // 2
            # Count subarrays with at most mid distinct elements
            count = countSubarraysWithAtMostKDistinct(mid)
            
            # If count >= (total + 1) // 2, then mid could be the median
            if count >= (total + 1) // 2:
                right = mid
            else:
                left = mid + 1
        
        return left