class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        covered_points = set()
        
        # For each car, add all points it covers to the set
        for start, end in nums:
            for point in range(start, end + 1):
                covered_points.add(point)
        
        return len(covered_points)