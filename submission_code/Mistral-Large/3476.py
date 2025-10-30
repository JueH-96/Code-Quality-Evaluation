from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        # Function to calculate the number of operations needed to make a number divisible by 3
        def operations_to_divisible_by_3(n: int) -> int:
            if n % 3 == 0:
                return 0
            elif n % 3 == 1:
                return 1
            else:
                return 1

        total_operations = 0
        for num in nums:
            total_operations += operations_to_divisible_by_3(num)

        return total_operations