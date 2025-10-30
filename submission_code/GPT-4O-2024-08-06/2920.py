from typing import List
from collections import defaultdict

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
        
        # Dictionary to store the last seen index of each number
        last_seen = defaultdict(lambda: -1)
        # Dictionary to store the maximum distance between consecutive occurrences of each number
        max_distance = defaultdict(int)
        
        # Iterate over the array to calculate the maximum distance for each number
        for i in range(n):
            num = nums[i]
            if last_seen[num] != -1:
                # Calculate the distance between current and last occurrence
                distance = i - last_seen[num]
                max_distance[num] = max(max_distance[num], distance)
            last_seen[num] = i
        
        # We need to consider the circular nature of the array
        # Calculate the distance from the last occurrence to the first occurrence
        for num in last_seen:
            distance = n - last_seen[num] + nums.index(num)
            max_distance[num] = max(max_distance[num], distance)
        
        # The minimum number of seconds needed is half of the maximum distance (rounded up)
        # because we can change both ends of the segment simultaneously
        min_seconds = float('inf')
        for num in max_distance:
            min_seconds = min(min_seconds, (max_distance[num] + 1) // 2)
        
        return min_seconds