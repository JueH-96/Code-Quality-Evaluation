class Solution:
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        # Calculate the sum of the digits of x
        digit_sum = sum(int(digit) for digit in str(x))
        # Check if x is divisible by the sum of its digits
        if x % digit_sum == 0:
            return digit_sum
        else:
            return -1

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.sumOfTheDigitsOfHarshadNumber(18))  # Expected output: 9
    print(sol.sumOfTheDigitsOfHarshadNumber(23))  # Expected output: -1