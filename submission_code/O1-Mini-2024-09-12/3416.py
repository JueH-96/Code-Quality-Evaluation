class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        n = len(nums)
        num = nums[0]
        D = 0
        temp = num
        while temp > 0:
            D += 1
            temp //= 10
        counts = [[0] * 10 for _ in range(D)]
        for num in nums:
            temp = num
            for pos in range(D):
                digit = temp % 10
                counts[pos][digit] += 1
                temp //= 10
        total = 0
        for pos in range(D):
            for d in range(10):
                cnt = counts[pos][d]
                total += cnt * (n - cnt)
        return total // 2