from collections import defaultdict

class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        # Determine the number of digits in each number
        num_str = str(nums[0])
        len_digits = len(num_str)
        
        # Create a list to hold digits for each position
        digits_per_pos = [[] for _ in range(len_digits)]
        
        for num in nums:
            s = str(num)
            for i in range(len_digits):
                digits_per_pos[i].append(s[i])
        
        total = 0
        n = len(nums)
        total_pairs = n * (n - 1) // 2
        
        for pos in range(len_digits):
            digits = digits_per_pos[pos]
            freq = defaultdict(int)
            for d in digits:
                freq[d] += 1
            
            same = 0
            for cnt in freq.values():
                same += cnt * (cnt - 1) // 2
            
            differing = total_pairs - same
            total += differing
        
        return total