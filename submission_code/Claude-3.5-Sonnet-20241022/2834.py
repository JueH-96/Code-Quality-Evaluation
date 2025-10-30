class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        positions = set(nums)
        
        for i in range(len(moveFrom)):
            if moveFrom[i] in positions:
                positions.remove(moveFrom[i])
                positions.add(moveTo[i])
                
        return sorted(list(positions))