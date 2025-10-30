class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        covered_points = set()
        for interval in nums:
            start, end = interval
            covered_points.update(range(start, end + 1))
        return len(covered_points)