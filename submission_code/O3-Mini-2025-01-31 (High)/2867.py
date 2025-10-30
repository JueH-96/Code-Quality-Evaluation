from typing import List

class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        
        # Collect the indices where the element 1 appears.
        ones_positions = [i for i, value in enumerate(nums) if value == 1]
        
        # If there are no 1's, it's impossible to split the array 
        # into subarrays each containing exactly one 1.
        if not ones_positions:
            return 0
        
        # If there is exactly one 1, the whole array is one good subarray.
        ways = 1
        # For every two adjacent 1's, the gap between them offers several splitting options.
        # Specifically, if pos[i-1] and pos[i] are adjacent 1 indices, there are (pos[i] - pos[i-1]) ways 
        # to decide the boundary between the subarrays.
        for i in range(1, len(ones_positions)):
            gap = ones_positions[i] - ones_positions[i - 1]
            ways = (ways * gap) % mod
        
        return ways