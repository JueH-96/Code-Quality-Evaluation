from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        # Map from maximum digit to the two largest numbers having that max digit
        top_two = {}  # max_digit -> [first_max, second_max]
        
        for num in nums:
            # Compute the maximum digit in num
            m = 0
            x = num
            while x > 0:
                m = max(m, x % 10)
                x //= 10
            
            if m not in top_two:
                top_two[m] = [num, -1]
            else:
                first, second = top_two[m]
                if num > first:
                    top_two[m] = [num, first]
                elif num > second:
                    top_two[m][1] = num
        
        # Now find the maximum sum of any two valid numbers
        ans = -1
        for first, second in top_two.values():
            if second != -1:
                ans = max(ans, first + second)
        
        return ans