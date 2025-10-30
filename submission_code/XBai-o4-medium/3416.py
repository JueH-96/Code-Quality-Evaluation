class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        nums_str = [str(num) for num in nums]
        m = len(nums_str[0])
        n = len(nums)
        total = 0
        
        for pos in range(m):
            counts = [0] * 10
            for s in nums_str:
                digit = int(s[pos])
                counts[digit] += 1
            
            total_pairs = n * (n - 1) // 2
            sum_same = 0
            for c in counts:
                sum_same += c * (c - 1) // 2
            
            contribution = total_pairs - sum_same
            total += contribution
        
        return total