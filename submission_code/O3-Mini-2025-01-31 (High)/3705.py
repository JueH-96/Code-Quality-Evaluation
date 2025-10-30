from typing import List
from collections import defaultdict

class Solution:
    def largestInteger(self, nums: List[int], k: int) -> int:
        # Dictionary to count how many subarrays each integer appears in.
        freq = defaultdict(int)
        n = len(nums)
        
        # Loop through each subarray of size k and update the frequency.
        for i in range(n - k + 1):
            # Use a set to count each number only once per subarray.
            unique_nums = set(nums[i:i+k])
            for num in unique_nums:
                freq[num] += 1
        
        # Find the largest integer that appears exactly once in some subarray.
        result = -1
        for num, count in freq.items():
            if count == 1:
                result = max(result, num)
                
        return result