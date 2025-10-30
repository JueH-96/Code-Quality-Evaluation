class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        covered_points = set()
        for start, end in nums:
            covered_points.update(range(start, end + 1))
        return len(covered_points)