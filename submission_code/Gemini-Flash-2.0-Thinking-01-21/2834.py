from collections import Counter

class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        pos_counts = Counter(nums)
        for i in range(len(moveFrom)):
            from_pos = moveFrom[i]
            to_pos = moveTo[i]
            count = pos_counts.get(from_pos, 0)
            if count > 0:
                pos_counts[to_pos] += count
                pos_counts[from_pos] = 0
        
        occupied_positions = []
        for pos, count in pos_counts.items():
            if count > 0:
                occupied_positions.append(pos)
        
        occupied_positions.sort()
        return occupied_positions