from typing import List
from collections import defaultdict

class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        n = len(nums)
        total_operations = float('inf')
        
        # We will use a sliding window approach to calculate the cost for each segment of size x
        for start in range(n - x + 1):
            # Calculate the frequency of elements in the current window of size x
            freq = defaultdict(int)
            for i in range(start, start + x):
                freq[nums[i]] += 1
            
            # We need to find the minimum operations to make all elements in this window equal
            # We will consider each unique number in the frequency map as a target
            for target in freq.keys():
                operations = 0
                for num in freq:
                    operations += abs(num - target) * freq[num]
                
                # Calculate the total operations needed for k segments
                total_operations = min(total_operations, operations * k)
        
        return total_operations

# Example usage:
# sol = Solution()
# print(sol.minOperations([5,-2,1,3,7,3,6,4,-1], 3, 2))  # Output: 8
# print(sol.minOperations([9,-2,-2,-2,1,5], 2, 2))      # Output: 3