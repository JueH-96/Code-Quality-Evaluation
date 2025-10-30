class Solution:
    def numberOfPoints(self, nums):
        points = set()
        for car in nums:
            for point in range(car[0], car[1] + 1):
                points.add(point)
        return len(points)