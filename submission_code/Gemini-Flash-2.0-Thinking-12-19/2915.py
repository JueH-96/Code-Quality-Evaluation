class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        n = len(nums)
        interesting_subarray_count = 0
        for l in range(n):
            for r in range(l, n):
                cnt = 0
                for i in range(l, r + 1):
                    if nums[i] % modulo == k:
                        cnt += 1
                if cnt % modulo == k:
                    interesting_subarray_count += 1
        return interesting_subarray_count