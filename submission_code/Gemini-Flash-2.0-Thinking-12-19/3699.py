class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        for p in range(n - 3):
            for q in range(p + 2, n - 1):
                for r in range(q + 2, n):
                    for s in range(r + 2, n + 1):
                        if s > n:
                            continue
                        if q - p > 1 and r - q > 1 and s - r > 1:
                            if nums[p] * nums[r] == nums[q] * nums[s-1]:
                                count += 1
        return count