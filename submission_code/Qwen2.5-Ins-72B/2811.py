class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        # Initialize the sum and the current number
        total_sum = 0
        current_number = 1
        used_numbers = set()
        
        # Loop until we have n numbers in the k-avoiding array
        while len(used_numbers) < n:
            # Check if the current number can be added without violating the k-avoiding condition
            if k - current_number not in used_numbers:
                used_numbers.add(current_number)
                total_sum += current_number
            current_number += 1
        
        return total_sum