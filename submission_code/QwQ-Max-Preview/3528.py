from collections import deque
from typing import List

class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
        
        dp = [0] * n
        dp[-1] = 0  # The last index has no further jumps
        
        # Deque to maintain lines for the convex hull trick, stored as (a, b) where the line is a*x + b
        dq = deque()
        dq.append((n-1, dp[-1]))
        
        for i in range(n-2, -1, -1):
            current_x = nums[i]
            best_val = -float('inf')
            
            # Binary search to find the optimal line in the deque
            left = 0
            right = len(dq) - 1
            best_idx = 0
            
            while left <= right:
                mid = (left + right) // 2
                a, b = dq[mid]
                current_val = a * current_x + b
                
                if current_val > best_val:
                    best_val = current_val
                    best_idx = mid
                
                # Check mid+1 if exists
                if mid < len(dq) - 1:
                    a_next, b_next = dq[mid + 1]
                    next_val = a_next * current_x + b_next
                    if next_val > best_val:
                        best_val = next_val
                        best_idx = mid + 1
                        left = mid + 1
                        right = len(dq) - 1
                        continue
                
                # Check mid-1 if exists
                if mid > 0:
                    a_prev, b_prev = dq[mid - 1]
                    prev_val = a_prev * current_x + b_prev
                    if prev_val > best_val:
                        best_val = prev_val
                        best_idx = mid - 1
                        right = mid - 1
                        continue
                
                # No better found, break
                break
            
            dp[i] = best_val - i * current_x
            
            # Add the current line (i, dp[i]) to the deque, maintaining the convex hull
            new_line = (i, dp[i])
            while len(dq) >= 2:
                a1, b1 = dq[-2]
                a2, b2 = dq[-1]
                
                # Calculate intersection x between the last two lines in the deque
                if a1 == a2:
                    if new_line[1] >= b2:
                        dq.pop()
                    else:
                        break
                    continue
                
                numerator = (b2 - b1)
                denominator = (a1 - a2)
                x_intersect = numerator / denominator
                
                # Calculate the value of the new line and the last line at x_intersect
                val_new = new_line[0] * x_intersect + new_line[1]
                val_last = a2 * x_intersect + b2
                
                if val_new >= val_last - 1e-9:
                    dq.pop()
                else:
                    break
            
            dq.append(new_line)
        
        return dp[0]