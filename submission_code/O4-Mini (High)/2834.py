class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        # Build a count map of current marble positions
        counts = {}
        for x in nums:
            counts[x] = counts.get(x, 0) + 1
        
        # Perform each move: remove all marbles from moveFrom[i] and add them to moveTo[i]
        for f, t in zip(moveFrom, moveTo):
            # Pop the count at 'f'; if none, pop returns 0
            cnt = counts.pop(f, 0)
            if cnt:
                counts[t] = counts.get(t, 0) + cnt
        
        # The remaining keys in counts are the occupied positions
        return sorted(counts.keys())