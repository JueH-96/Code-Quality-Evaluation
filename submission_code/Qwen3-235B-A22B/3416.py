from typing import List

class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return 0
        
        # Precompute the digits of each number as a list of characters
        digits_list = []
        # Determine the number of digits from the first number
        s = str(nums[0])
        k = len(s)
        for num in nums:
            s = str(num)
            digits_list.append(list(s))  # each element is a list of characters
        
        total_pairs = n * (n - 1) // 2
        total = 0
        
        for i in range(k):
            # Count frequency of each digit in position i
            counts = [0] * 10
            for j in range(n):
                d = int(digits_list[j][i])
                counts[d] += 1
            # Calculate sum_same
            sum_same = 0
            for cnt in counts:
                sum_same += cnt * (cnt - 1) // 2
            contribution = total_pairs - sum_same
            total += contribution
        
        return total