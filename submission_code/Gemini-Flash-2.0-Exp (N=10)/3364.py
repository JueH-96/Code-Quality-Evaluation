from typing import List

class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n = len(nums)
        m = len(andValues)
        
        if m > n:
            return -1

        dp = {}

        def solve(index, and_index):
            if (index, and_index) in dp:
                return dp[(index, and_index)]
            
            if and_index == m:
                if index == n:
                    return 0
                else:
                    return float('inf')
            
            if index == n:
                return float('inf')

            min_sum = float('inf')
            current_and = nums[index]
            for i in range(index, n):
                current_and &= nums[i]
                if current_and == andValues[and_index]:
                    next_sum = solve(i + 1, and_index + 1)
                    if next_sum != float('inf'):
                        min_sum = min(min_sum, nums[i] + next_sum)
            
            dp[(index, and_index)] = min_sum
            return min_sum

        result = solve(0, 0)
        return result if result != float('inf') else -1