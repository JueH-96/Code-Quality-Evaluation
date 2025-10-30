from collections import defaultdict

class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        # Initialize a frequency map to count marbles at each position
        freq = defaultdict(int)
        for num in nums:
            freq[num] += 1
        
        # Perform the moves
        for i in range(len(moveFrom)):
            from_pos = moveFrom[i]
            to_pos = moveTo[i]
            if from_pos in freq:
                count = freq[from_pos]
                del freq[from_pos]
                freq[to_pos] += count
        
        # Extract the occupied positions and sort them
        occupied = sorted(freq.keys())
        return occupied