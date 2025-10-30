class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        covered_points = set()
        for start, end in nums:
            for i in range(start, end + 1):
                covered_points.add(i)
        return len(covered_points)