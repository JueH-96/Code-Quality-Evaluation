class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        decrement_count = [0] * n
        for query in queries:
            l, r = query
            for i in range(l, r + 1):
                decrement_count[i] += 1

        for i in range(n):
            if decrement_count[i] < nums[i]:
                return False
        return True