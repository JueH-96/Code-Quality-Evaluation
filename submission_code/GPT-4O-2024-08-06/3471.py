from typing import List

class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        averages = []
        
        # Sort the nums array to easily access min and max elements
        nums.sort()
        
        # Repeat the process n/2 times
        while nums:
            # Remove the smallest and largest elements
            minElement = nums.pop(0)
            maxElement = nums.pop(-1)
            
            # Calculate the average and add to averages list
            average = (minElement + maxElement) / 2
            averages.append(average)
        
        # Return the minimum element in averages
        return min(averages)