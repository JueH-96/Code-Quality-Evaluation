from bisect import bisect_left, bisect_right

class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        # Create a list of differences for the given target
        diffs = [0] * len(nums)
        start_points = [[] for _ in range(len(nums))]
        
        for i, num in enumerate(nums[:-1]):
            valid_start_points = []
            for j in range(len(nums)-1, i, -1):
                if nums[j] - num <= target and nums[j] - num >= -target:
                    valid_start_points.append(j)
                    start_points[j].append(i)
                else:
                    break  # No need to check further if it's already out of range
            
            # No valid jump from current position, break the loop early
            if not valid_start_points:
                break
            
            diffs[i] = valid_start_points[0] if valid_start_points else -1
        
        # No valid jumps from any position, return -1
        if not start_points:
            return -1
        
        # Initialize memoization with -1
        memo = [-1] * len(nums)
        
        def dfs(index):
            if memo[index] != -1:
                return memo[index]  # Return cached value
            
            if index == len(nums) - 1:
                return 0  # Already at the last index
            
            max_jumps = -1
            for next_index in start_points[index]:
                jumps = dfs(next_index) + 1
                max_jumps = max(max_jumps, jumps)
            
            memo[index] = max_jumps  # Cache the result
            return max_jumps
        
        return dfs(0)