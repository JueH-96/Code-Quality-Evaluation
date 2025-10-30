class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        # Create a dictionary to store the count of marbles at each position
        marble_counts = {}
        for num in nums:
            marble_counts[num] = marble_counts.get(num, 0) + 1
        
        # Apply the moves and update the marble counts
        for i in range(len(moveFrom)):
            from_pos = moveFrom[i]
            to_pos = moveTo[i]
            
            # Decrement the count at the 'from' position
            marble_counts[from_pos] -= 1
            
            # Increment the count at the 'to' position
            marble_counts[to_pos] = marble_counts.get(to_pos, 0) + 1
            
            # Remove the position if the count becomes 0
            if marble_counts[from_pos] == 0:
                del marble_counts[from_pos]
        
        # Return the sorted list of occupied positions
        return sorted(marble_counts.keys())