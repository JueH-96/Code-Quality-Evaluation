class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        n = len(nums)
        count = 0
        for i in range(n):
            for j in range(i, n):
                cnt = 0
                for l in range(i, j + 1):
                    if nums[l] % modulo == k:
                        cnt += 1
                if cnt % modulo == k:
                    count += 1
        return count