from typing import List

class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n = len(nums)
        m = len(andValues)
        
        dp = {}
        
        def solve(index, and_index):
            if and_index == m:
                if index == n:
                    return 0
                else:
                    return float('inf')
            
            if index == n:
                return float('inf')
            
            if (index, and_index) in dp:
                return dp[(index, and_index)]
            
            ans = float('inf')
            current_and = nums[index]
            for j in range(index, n):
                current_and &= nums[j]
                if current_and == andValues[and_index]:
                    next_val = solve(j + 1, and_index + 1)
                    if next_val != float('inf'):
                        ans = min(ans, nums[j] + next_val)
            
            dp[(index, and_index)] = ans
            return ans
        
        result = solve(0, 0)
        if result == float('inf'):
            return -1
        else:
            return result