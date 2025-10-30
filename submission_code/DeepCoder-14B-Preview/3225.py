from collections import defaultdict

class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        left = 0
        max_length = 0
        freq = defaultdict(int)
        violations = 0
        
        for right in range(len(nums)):
            element = nums[right]
            current_count = freq[element]
            
            if current_count == k:
                violations += 1
            freq[element] += 1
            
            while violations > 0:
                left_element = nums[left]
                current_left_count = freq[left_element]
                
                if current_left_count > k:
                    if (current_left_count - 1) <= k:
                        violations -= 1
                freq[left_element] -= 1
                if freq[left_element] == 0:
                    del freq[left_element]
                left += 1
            
            current_window = right - left + 1
            if current_window > max_length:
                max_length = current_window
        
        return max_length