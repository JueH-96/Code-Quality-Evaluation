from collections import defaultdict

class Solution:
    def minimumIndex(self, nums: list[int]) -> int:
        n = len(nums)
        if n < 2:
            return -1
        
        # Determine the dominant element
        freq = defaultdict(int)
        for num in nums:
            freq[num] += 1
        
        dominant = None
        for key in freq:
            if freq[key] * 2 > n:
                dominant = key
                break
        
        # Build the prefix sum array for the dominant element
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + (1 if nums[i] == dominant else 0)
        
        total_dom = prefix[n]
        
        # Check each possible split
        for i in range(n - 1):
            left_count = prefix[i + 1]
            left_length = i + 1
            if left_count * 2 <= left_length:
                continue
            
            right_count = total_dom - left_count
            right_length = n - i - 1
            if right_count * 2 <= right_length:
                continue
            
            return i
        
        return -1