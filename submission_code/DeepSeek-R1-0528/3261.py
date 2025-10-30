class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ans = 0
        for b in range(30):
            count_zeros = 0
            for num in nums:
                if (num >> b) & 1 == 0:
                    count_zeros += 1
            if count_zeros < n - k:
                ans |= (1 << b)
        return ans