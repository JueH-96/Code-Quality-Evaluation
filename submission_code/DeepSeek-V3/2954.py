from collections import defaultdict

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        n = len(nums)
        if n < k:
            return 0
        
        max_sum = 0
        current_sum = sum(nums[:k])
        freq = defaultdict(int)
        for num in nums[:k]:
            freq[num] += 1
        
        if len(freq) >= m:
            max_sum = current_sum
        
        for i in range(k, n):
            # Remove the leftmost element of the previous window
            left_num = nums[i - k]
            current_sum -= left_num
            freq[left_num] -= 1
            if freq[left_num] == 0:
                del freq[left_num]
            
            # Add the new element to the window
            right_num = nums[i]
            current_sum += right_num
            freq[right_num] += 1
            
            # Check if the current window is almost unique
            if len(freq) >= m:
                max_sum = max(max_sum, current_sum)
        
        return max_sum