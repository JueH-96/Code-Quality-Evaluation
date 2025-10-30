from typing import List

class Solution:
    def countPairs(self, nums: List[int], target: int) -> int:
        n = len(nums)
        count = 0
        
        # Iterate through all possible first indices 'i'.
        # 'i' can go from 0 up to n-2. 
        # If 'i' is n-1, there would be no 'j' such that j > i and j < n.
        for i in range(n - 1): 
            # Iterate through all possible second indices 'j'.
            # 'j' must be greater than 'i' and less than 'n'.
            for j in range(i + 1, n):
                # Check if the sum of elements at indices i and j is less than the target.
                if nums[i] + nums[j] < target:
                    # If the condition is met, increment the count.
                    count += 1
                    
        # Return the total count of pairs that satisfy the conditions.
        return count