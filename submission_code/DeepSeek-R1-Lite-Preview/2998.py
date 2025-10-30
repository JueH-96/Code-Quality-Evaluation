class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0  # Initialize counter for symmetric numbers
        for num in range(low, high + 1):
            s = str(num)  # Convert number to string
            length = len(s)
            if length % 2 == 0:  # Check for even number of digits
                n = length // 2
                first_half = s[:n]
                second_half = s[n:]
                sum_first = sum(int(digit) for digit in first_half)
                sum_second = sum(int(digit) for digit in second_half)
                if sum_first == sum_second:
                    count += 1  # Increment counter for symmetric numbers
        return count