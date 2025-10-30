class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        covered = set()
        for interval in nums:
            start, end = interval
            for point in range(start, end + 1):
                covered.add(point)
        return len(covered)