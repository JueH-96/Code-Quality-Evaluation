from typing import List

class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Precompute max and min for all ranges [i][j]
        max_between = [[0] * n for _ in range(n)]
        min_between = [[0] * n for _ in range(n)]
        
        for i in range(n):
            max_between[i][i] = nums[i]
            min_between[i][i] = nums[i]
            for j in range(i + 1, n):
                max_between[i][j] = max(max_between[i][j - 1], nums[j])
                min_between[i][j] = min(min_between[i][j - 1], nums[j])
        
        total = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if nums[i] >= nums[j]:
                    continue
                if nums[j] - nums[i] <= 1:
                    continue
                
                a = i + 1
                b = j - 1
                valid = True
                if a <= b:
                    max_val = max_between[a][b]
                    min_val = min_between[a][b]
                    if not (max_val <= nums[i] or min_val >= nums[j]):
                        valid = False
                
                if valid:
                    count = (i + 1) * (n - j)
                    total += count
        
        return total