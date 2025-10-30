class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        line = [0] * 101
        for start, end in nums:
            for i in range(start, end + 1):
                line[i] = 1
        return sum(line)