class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        covered = set()
        for interval in nums:
            start, end = interval[0], interval[1]
            for i in range(start, end + 1):
                covered.add(i)
        return len(covered)