from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        max_digit_pairs = {}
        
        for num in nums:
            max_digit = max(str(num))
            if max_digit in max_digit_pairs:
                max_digit_pairs[max_digit].append(num)
            else:
                max_digit_pairs[max_digit] = [num]
        
        max_sum = -1
        for pairs in max_digit_pairs.values():
            if len(pairs) > 1:
                pairs.sort(reverse=True)
                max_sum = max(max_sum, pairs[0] + pairs[1])
        
        return max_sum