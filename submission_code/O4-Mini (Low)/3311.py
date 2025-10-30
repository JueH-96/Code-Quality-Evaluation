from typing import List

class Solution:
    def returnToBoundaryCount(self, nums: List[int]) -> int:
        # Initialize the ant's position at the boundary (0)
        pos = 0
        # Counter for the number of times the ant returns to the boundary
        count = 0
        
        # Process each move in nums
        for step in nums:
            # Move the ant
            pos += step
            # Check if it's exactly on the boundary after the move
            if pos == 0:
                count += 1
        
        return count

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.returnToBoundaryCount([2, 3, -5]))   # Expected output: 1
    print(sol.returnToBoundaryCount([3, 2, -3, -4]))  # Expected output: 0