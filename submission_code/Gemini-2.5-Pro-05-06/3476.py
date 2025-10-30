from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        operations_count = 0
        for num in nums:
            # If a number is not divisible by 3, we need to adjust it.
            # The goal is to reach the nearest multiple of 3.
            
            # If num % 3 == 1:
            #   Subtracting 1 makes it divisible by 3 (e.g., num=4 becomes 4-1=3). This is 1 operation.
            #   (Adding 2 would also work (e.g., num=1 becomes 1+2=3), but that's 2 operations.)
            #   So, 1 operation is optimal.
            
            # If num % 3 == 2:
            #   Adding 1 makes it divisible by 3 (e.g., num=2 becomes 2+1=3; num=5 becomes 5+1=6). This is 1 operation.
            #   (Subtracting 2 would also work (e.g., num=5 becomes 5-2=3), but that's 2 operations.)
            #   So, 1 operation is optimal.
            
            # In summary, if num % 3 is not 0 (i.e., it's 1 or 2), 
            # one operation is sufficient and necessary.
            if num % 3 != 0:
                operations_count += 1
                
        return operations_count