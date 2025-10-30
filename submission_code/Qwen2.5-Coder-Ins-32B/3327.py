from typing import List
from collections import deque

class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        ones = [i for i in range(n) if nums[i] == 1]
        
        if len(ones) >= k:
            # If there are enough ones, we can use a sliding window to find the minimum cost
            min_cost = float('inf')
            window = deque()
            cost = 0
            
            for i, one in enumerate(ones):
                window.append(one)
                if i % 2 == 0:
                    cost += one - ones[i // 2]
                else:
                    cost += ones[(i + 1) // 2] - one
                
                if len(window) > k:
                    if (len(window) - 1) % 2 == 0:
                        cost -= ones[(len(window) - 2) // 2] - window.popleft()
                    else:
                        cost -= window.popleft() - ones[(len(window) - 1) // 2]
                
                if len(window) == k:
                    min_cost = min(min_cost, cost)
            
            return min_cost
        
        # If there are not enough ones, we need to use maxChanges
        # We will try to place ones optimally using maxChanges
        # and then use the sliding window to find the minimum cost
        
        # First, we place ones using maxChanges
        placed_ones = []
        for i in range(n):
            if nums[i] == 0 and maxChanges > 0:
                placed_ones.append(i)
                maxChanges -= 1
        
        # Combine the original ones and the placed ones
        all_ones = sorted(ones + placed_ones)
        
        # Now, we use the sliding window to find the minimum cost
        min_cost = float('inf')
        window = deque()
        cost = 0
        
        for i, one in enumerate(all_ones):
            window.append(one)
            if i % 2 == 0:
                cost += one - all_ones[i // 2]
            else:
                cost += all_ones[(i + 1) // 2] - one
            
            if len(window) > k:
                if (len(window) - 1) % 2 == 0:
                    cost -= all_ones[(len(window) - 2) // 2] - window.popleft()
                else:
                    cost -= window.popleft() - all_ones[(len(window) - 1) // 2]
            
            if len(window) == k:
                min_cost = min(min_cost, cost)
        
        return min_cost