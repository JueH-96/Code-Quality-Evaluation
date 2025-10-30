from typing import List

class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        # Sort the maximum heights in descending order
        maximumHeight.sort(reverse=True)
        
        total_sum = 0
        # Initially, we can assign any height (start with a huge number)
        prev_assigned = float('inf')
        
        for height in maximumHeight:
            # The current tower's assigned height can be at most:
            # 1. Its maximum allowed height 'height'
            # 2. Or one less than the previous assigned height (to keep distinct and maximize sum)
            # We take the minimum of these two.
            current = min(height, prev_assigned - 1)
            
            # If current becomes less than 1, then it's not possible to assign a valid positive integer height.
            if current < 1:
                return -1
            
            total_sum += current
            prev_assigned = current  # update for the next iteration
            
        return total_sum

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.maximumTotalSum([2,3,4,3]))  # Expected output: 10
    print(sol.maximumTotalSum([15,10]))    # Expected output: 25
    print(sol.maximumTotalSum([2,2,1]))     # Expected output: -1