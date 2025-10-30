class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        points = set()
        for start, end in nums:
            for i in range(start, end+1):
                points.add(i)
        return len(points)