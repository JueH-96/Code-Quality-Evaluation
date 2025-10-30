class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        covered = set()
        for start, end in nums:
            for i in range(start, end + 1):
                covered.add(i)
        return len(covered)