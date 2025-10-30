class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        # Convert nums to a set for O(1) average time complexity for lookups and deletions
        positions = set(nums)

        # Process each move
        for from_pos, to_pos in zip(moveFrom, moveTo):
            if from_pos in positions:
                positions.remove(from_pos)
            positions.add(to_pos)

        # Convert the set back to a sorted list and return
        return sorted(positions)