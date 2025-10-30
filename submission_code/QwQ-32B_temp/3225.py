from typing import List

class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        freq = {}
        max_len = 0
        left = 0
        for right in range(len(nums)):
            num = nums[right]
            freq[num] = freq.get(num, 0) + 1
            
            # If the current number's frequency exceeds k, move left pointer to reduce
            while freq[num] > k:
                left_num = nums[left]
                freq[left_num] -= 1
                if freq[left_num] == 0:
                    del freq[left_num]
                left += 1
            
            # Update the maximum length of valid subarray
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
        
        return max_len