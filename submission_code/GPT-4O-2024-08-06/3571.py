from typing import List
from bisect import bisect_left

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        # Sort the coordinates based on x first, then y
        coordinates.sort()
        
        # Extract the y-coordinates
        y_coords = [y for _, y in coordinates]
        
        # Find the index of the y-coordinate of the point at index k
        target_y = coordinates[k][1]
        
        # Initialize a list to store the longest increasing subsequence of y-coordinates
        lis = []
        
        # Iterate over the y-coordinates
        for y in y_coords:
            # Find the position to replace or append in the lis
            pos = bisect_left(lis, y)
            if pos < len(lis):
                lis[pos] = y
            else:
                lis.append(y)
            
            # If we have reached the y-coordinate of the target point, break
            if y == target_y:
                break
        
        # The length of the lis up to the target point is the answer
        return len(lis)