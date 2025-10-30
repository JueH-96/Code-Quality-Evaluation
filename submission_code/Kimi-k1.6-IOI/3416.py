class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        d = len(str(nums[0]))
        counts = [[0] * 10 for _ in range(d)]
        
        for num in nums:
            s = str(num)
            for i in range(d):
                digit = int(s[i])
                counts[i][digit] += 1
        
        total = 0
        for pos in range(d):
            sum_same = 0
            for c in counts[pos]:
                sum_same += c * (c - 1) // 2
            total_pairs = n * (n - 1) // 2
            diff_pairs = total_pairs - sum_same
            total += diff_pairs
        
        return total