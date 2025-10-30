import heapq

class Solution:
    def maximumJumps(self, nums: list[int], target: int) -> int:
        n = len(nums)
        if n == 0:
            return -1
        
        # Initialize the dp array with -infinity, except for the starting index
        dp = [-float('inf')] * n
        dp[0] = 0
        
        # Use a max-heap by pushing negative values
        heap = []
        heapq.heappush(heap, (0, 0))  # (negative_jumps, index), since we use min-heap as max-heap
        
        while heap:
            current_jumps_neg, i = heapq.heappop(heap)
            current_jumps = -current_jumps_neg
            
            # If a better path to i has already been found, skip this state
            if current_jumps < dp[i]:
                continue
            
            # If we've reached the last index, return the number of jumps
            if i == n - 1:
                return current_jumps
            
            # Explore all possible next indices
            for j in range(i + 1, n):
                diff = nums[j] - nums[i]
                if -target <= diff <= target:
                    new_jumps = current_jumps + 1
                    if new_jumps > dp[j]:
                        dp[j] = new_jumps
                        heapq.heappush(heap, (-new_jumps, j))
        
        # If the last index was never reached
        return -1 if dp[-1] == -float('inf') else dp[-1]