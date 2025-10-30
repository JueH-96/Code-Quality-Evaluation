class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        freq = {}
        max_length = 0
        left = 0
        for right in range(len(nums)):
            num = nums[right]
            freq[num] = freq.get(num, 0) + 1
            
            # If the current element's frequency exceeds k, move left to reduce the window size
            while freq[num] > k:
                left_num = nums[left]
                freq[left_num] -= 1
                if freq[left_num] == 0:
                    del freq[left_num]
                left += 1
            
            # Update the maximum length of the valid subarray
            current_length = right - left + 1
            if current_length > max_length:
                max_length = current_length
        
        return max_length