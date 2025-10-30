class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        needed_decrements = list(nums)

        for l, r in queries:
            for i in range(l, r + 1):
                if needed_decrements[i] > 0:
                    needed_decrements[i] -= 1

        return all(nd == 0 for nd in needed_decrements)