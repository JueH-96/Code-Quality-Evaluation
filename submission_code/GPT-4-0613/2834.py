class Solution:
    def relocateMarbles(self, nums, moveFrom, moveTo):
        from collections import defaultdict
        marble_positions = defaultdict(int)
        for num in nums:
            marble_positions[num] += 1
        for i in range(len(moveFrom)):
            if marble_positions[moveFrom[i]] > 0:
                marble_positions[moveTo[i]] += marble_positions[moveFrom[i]]
                marble_positions[moveFrom[i]] = 0
        return sorted([key for key in marble_positions.keys() if marble_positions[key] > 0])