class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        for a in range(n - 2 * k + 1):
            first_incr = all(nums[a + i] < nums[a + i + 1] for i in range(k - 1))
            if not first_incr:
                continue
            b = a + k
            second_incr = all(nums[b + i] < nums[b + i + 1] for i in range(k - 1))
            if second_incr:
                return True
        return False