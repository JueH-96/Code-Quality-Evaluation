class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        
        freq = defaultdict(int)
        left = 0
        answer = 0
        
        for right in range(len(nums)):
            freq[nums[right]] += 1
            
            # If the frequency of the current element exceeds k, shrink the window
            while freq[nums[right]] > k:
                freq[nums[left]] -= 1
                if freq[nums[left]] == 0:
                    del freq[nums[left]]
                left += 1
            
            # Update the maximum length
            answer = max(answer, right - left + 1)
        
        return answer