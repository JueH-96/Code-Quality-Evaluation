class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        # Initialize the sum and the set to keep track of used numbers
        total_sum = 0
        used_numbers = set()

        # Start with the smallest positive integer
        current_number = 1

        while len(used_numbers) < n:
            # Check if the current number can be used
            if k % 2 == 0 and current_number == k // 2:
                current_number += 1
                continue

            if current_number not in used_numbers and (k - current_number) not in used_numbers:
                used_numbers.add(current_number)
                total_sum += current_number

            current_number += 1

        return total_sum