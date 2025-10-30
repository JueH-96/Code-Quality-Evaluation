class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        for p in range(n):
            for q in range(p+2, n):
                for r in range(q+2, n):
                    for s in range(r+2, n):
                        if nums[p] * nums[r] == nums[q] * nums[s]:
                            count += 1
        return count