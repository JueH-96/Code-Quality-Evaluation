class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        occupied = set(nums)
        for i in range(len(moveFrom)):
            if moveFrom[i] in occupied:
                occupied.remove(moveFrom[i])
                occupied.add(moveTo[i])
        return sorted(occupied)