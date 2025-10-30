class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7
        
        # Helper function to calculate the sum of digits of a number
        def digit_sum(num):
            return sum(int(digit) for digit in str(num))
        
        # Helper function to count good numbers in range [1, num]
        def count_good_numbers(num):
            count = 0
            for x in range(1, num + 1):
                if min_sum <= digit_sum(x) <= max_sum:
                    count += 1
            return count
        
        # Convert num1 and num2 to integers
        num1_int = int(num1)
        num2_int = int(num2)
        
        # Count good numbers in the range [1, num2] and subtract the count of good numbers in the range [1, num1 - 1]
        total_good_numbers = count_good_numbers(num2_int) - count_good_numbers(num1_int - 1)
        
        # Return the result modulo 10^9 + 7
        return total_good_numbers % MOD

# Example usage:
# sol = Solution()
# print(sol.count("1", "12", 1, 8))  # Output: 11
# print(sol.count("1", "5", 1, 5))   # Output: 5