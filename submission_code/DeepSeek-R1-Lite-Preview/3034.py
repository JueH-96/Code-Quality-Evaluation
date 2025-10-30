class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        points = set()
        for car in nums:
            start, end = car
            for point in range(start, end + 1):
                points.add(point)
        return len(points)