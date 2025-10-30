class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        for p in range(n - 6):
            for q in range(p + 2, n - 4):
                for r in range(q + 2, n - 2):
                    for s in range(r + 2, n):
                        if nums[p] * nums[r] == nums[q] * nums[s]:
                            count += 1
        return count