class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        from collections import defaultdict
        
        # Count how many marbles are at each position initially
        counts = defaultdict(int)
        for pos in nums:
            counts[pos] += 1
        
        # Process each move
        for f, t in zip(moveFrom, moveTo):
            counts[t] += counts[f]   # Move all marbles from position f to t
            counts[f] = 0           # Position f is no longer occupied after the move
        
        # Collect all positions that still have marbles, sort them, and return
        result = [pos for pos, c in counts.items() if c > 0]
        result.sort()
        return result