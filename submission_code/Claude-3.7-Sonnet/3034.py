class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        points_covered = set()
        
        for start, end in nums:
            for point in range(start, end + 1):
                points_covered.add(point)
        
        return len(points_covered)