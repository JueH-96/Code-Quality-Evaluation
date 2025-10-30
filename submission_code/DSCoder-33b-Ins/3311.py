class Solution:
    def returnToBoundaryCount(self, nums: List[int]) -> int:
        boundary_crossed = False
        boundary_cross_count = 0
        for num in nums:
            if num < 0:
                if boundary_crossed:
                    boundary_cross_count += 1
                boundary_crossed = False
            else:
                boundary_crossed = True
        return boundary_cross_count