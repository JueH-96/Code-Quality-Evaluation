from typing import List
import bisect

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        # Extract the target point
        x_k, y_k = coordinates[k]
        
        # Collect points for the left part (x < x_k and y < y_k)
        left_points = []
        for x, y in coordinates:
            if x < x_k and y < y_k:
                left_points.append((x, y))
        
        # Sort left points by x, then y
        left_points.sort()
        # Extract the y values for LIS calculation
        y_left = [y for x, y in left_points]
        # Compute the LIS for the left points' y values
        lis_left = self.length_of_lis(y_left)
        left_length = lis_left + 1  # include the current point
        
        # Collect points for the right part (x > x_k and y > y_k)
        right_points = []
        for x, y in coordinates:
            if x > x_k and y > y_k:
                right_points.append((x, y))
        
        # Sort right points by x, then y
        right_points.sort()
        # Extract the y values for LIS calculation
        y_right = [y for x, y in right_points]
        # Compute the LIS for the right points' y values
        lis_right = self.length_of_lis(y_right)
        right_length = lis_right + 1  # include the current point
        
        return left_length + right_length - 1  # subtract 1 to avoid double counting the current point
    
    def length_of_lis(self, nums: List[int]) -> int:
        tails = []
        for num in nums:
            # Find the index to replace in tails
            idx = bisect.bisect_left(tails, num)
            if idx == len(tails):
                tails.append(num)
            else:
                tails[idx] = num
        return len(tails)