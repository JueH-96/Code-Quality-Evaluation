from typing import List
from collections import deque

class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        # Create a queue for BFS, where each state is (nums, operations)
        queue = deque([(nums, 0)])
        # Create a set to store visited states
        visited = set(tuple(sorted(nums)))
        
        while queue:
            curr_nums, operations = queue.popleft()
            # Check if the current state contains a subsequence that sums to target
            if self.can_sum(curr_nums, target):
                return operations
            
            # Generate next states
            for i, num in enumerate(curr_nums):
                if num > 1:
                    next_nums = curr_nums[:i] + curr_nums[i+1:] + [num // 2, num // 2]
                    next_nums.sort()
                    next_tuple = tuple(next_nums)
                    if next_tuple not in visited:
                        queue.append((next_nums, operations + 1))
                        visited.add(next_tuple)
        
        # If no state contains a subsequence that sums to target, return -1
        return -1
    
    def can_sum(self, nums: List[int], target: int) -> bool:
        # Check if the current state contains a subsequence that sums to target
        dp = [False] * (target + 1)
        dp[0] = True
        for num in nums:
            for i in range(target, num - 1, -1):
                dp[i] = dp[i] or dp[i - num]
        return dp[target]