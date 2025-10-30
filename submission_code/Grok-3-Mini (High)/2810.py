import collections
from typing import List

class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        arr = nums + nums  # Double the array to handle cyclic nature
        ans = float('inf')
        
        for D in range(n):  # Iterate over possible maximum delays D
            W = D + 1  # Window size W = D + 1
            sum_swm = 0
            dq = collections.deque()
            
            for right in range(2 * n):  # Iterate over all positions in the doubled array
                # Remove indices that are out of the current window
                while dq and dq[0] < right - W + 1:
                    dq.popleft()
                
                # Remove indices with larger or equal values from the back
                while dq and arr[dq[-1]] >= arr[right]:
                    dq.pop()
                
                # Add current index to deque
                dq.append(right)
                
                # If the window is valid (size W), get the minimum value
                if right >= W - 1:
                    min_val = arr[dq[0]]
                    # Add to sum if the ending position corresponds to the first n elements
                    if n <= right <= 2 * n - 1:
                        sum_swm += min_val
            
            # Calculate the cost for this D and update the answer
            current_cost = sum_swm + x * D
            if current_cost < ans:
                ans = current_cost
        
        return ans