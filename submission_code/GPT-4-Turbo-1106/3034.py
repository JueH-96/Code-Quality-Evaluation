class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        covered_points = set()
        for car in nums:
            for point in range(car[0], car[1] + 1):
                covered_points.add(point)
        return len(covered_points)