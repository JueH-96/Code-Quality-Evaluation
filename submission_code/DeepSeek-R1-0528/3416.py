class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        n = len(nums)
        d = len(str(nums[0]))
        total_pairs = n * (n - 1) // 2
        num_strs = [str(x) for x in nums]
        result = 0
        
        for j in range(d):
            freq = [0] * 10
            for i in range(n):
                digit = int(num_strs[i][j])
                freq[digit] += 1
            same_pairs = 0
            for count in freq:
                same_pairs += count * (count - 1) // 2
            result += (total_pairs - same_pairs)
        
        return result