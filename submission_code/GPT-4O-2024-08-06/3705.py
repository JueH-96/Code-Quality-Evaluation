from typing import List
from collections import defaultdict

class Solution:
    def largestInteger(self, nums: List[int], k: int) -> int:
        # Dictionary to keep track of the count of each number in subarrays of size k
        count_in_subarrays = defaultdict(int)
        
        # Iterate over each subarray of size k
        for i in range(len(nums) - k + 1):
            subarray = nums[i:i+k]
            # Use a set to ensure we only count each number once per subarray
            unique_in_subarray = set(subarray)
            for num in unique_in_subarray:
                count_in_subarrays[num] += 1
        
        # Find the largest number that appears in exactly one subarray
        largest_almost_missing = -1
        for num, count in count_in_subarrays.items():
            if count == 1:
                largest_almost_missing = max(largest_almost_missing, num)
        
        return largest_almost_missing