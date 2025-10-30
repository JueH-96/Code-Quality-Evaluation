class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        left = 0
        max_length = 0
        freq = {}
        
        for right in range(len(nums)):
            # Add current element to window
            freq[nums[right]] = freq.get(nums[right], 0) + 1
            
            # Shrink window while any element frequency exceeds k
            while freq[nums[right]] > k:
                freq[nums[left]] -= 1
                if freq[nums[left]] == 0:
                    del freq[nums[left]]
                left += 1
            
            # Update maximum length
            max_length = max(max_length, right - left + 1)
        
        return max_length