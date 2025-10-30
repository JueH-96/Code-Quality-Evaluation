from typing import List

class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        # Sort the list in ascending order
        nums.sort()
        
        # Initialize an empty list to store the averages
        averages = []
        
        # Repeat the procedure n / 2 times
        while nums:
            # Remove the smallest element and the largest element from nums
            minElement = nums.pop(0)
            maxElement = nums.pop(-1)
            
            # Add (minElement + maxElement) / 2 to averages
            averages.append((minElement + maxElement) / 2)
        
        # Return the minimum element in averages
        return min(averages)