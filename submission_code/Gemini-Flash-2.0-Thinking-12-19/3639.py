class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        counts = [0] * n
        for i in range(n):
            for query in queries:
                l, r = query
                if l <= i <= r:
                    counts[i] += 1
        for i in range(n):
            if counts[i] < nums[i]:
                return False
        return True