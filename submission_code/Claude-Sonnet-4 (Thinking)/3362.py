class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        total_subarrays = n * (n + 1) // 2
        target = (total_subarrays + 1) // 2
        
        def count_subarrays_with_at_most_k_distinct(k):
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
        
        left, right = 1, len(set(nums))
        
        while left < right:
            mid = (left + right) // 2
            if count_subarrays_with_at_most_k_distinct(mid) >= target:
                right = mid
            else:
                left = mid + 1
        
        return left