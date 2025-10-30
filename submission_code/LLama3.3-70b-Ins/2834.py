from typing import List

class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        # Create a dictionary to store the count of marbles at each position
        marble_count = {}
        
        # Initialize the marble count dictionary with the initial positions
        for num in nums:
            if num in marble_count:
                marble_count[num] += 1
            else:
                marble_count[num] = 1
        
        # Apply the moves
        for i in range(len(moveFrom)):
            # Move the marbles from moveFrom[i] to moveTo[i]
            if moveFrom[i] in marble_count:
                marble_count[moveTo[i]] = marble_count.get(moveTo[i], 0) + marble_count[moveFrom[i]]
                del marble_count[moveFrom[i]]
        
        # Return the sorted list of occupied positions
        return sorted(marble_count.keys())