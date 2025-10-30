from collections import defaultdict
from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        max_digit_map = defaultdict(list)
        for num in nums:
            max_digit = max(int(d) for d in str(num))
            max_digit_map[max_digit].append(num)
        
        max_sum = -1
        for numbers in max_digit_map.values():
            if len(numbers) >= 2:
                numbers.sort(reverse=True)
                current_sum = numbers[0] + numbers[1]
                if current_sum > max_sum:
                    max_sum = current_sum
        return max_sum