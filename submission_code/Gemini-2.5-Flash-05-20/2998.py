class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        # Iterate through each number in the given range [low, high]
        for num in range(low, high + 1):
            s_num = str(num)  # Convert the number to a string to easily access its digits
            length = len(s_num)

            # Rule 1: Numbers with an odd number of digits are never symmetric
            if length % 2 != 0:
                continue

            # Rule 2: For an even number of digits (2 * n), determine n
            n = length // 2

            # Calculate the sum of the first n digits
            # Use sum(map(int, iterable)) for concise digit summing
            sum_first_half = sum(map(int, s_num[:n]))

            # Calculate the sum of the last n digits
            sum_second_half = sum(map(int, s_num[n:]))
            
            # Rule 3: Check if the two sums are equal
            if sum_first_half == sum_second_half:
                count += 1 # If symmetric, increment the counter
                
        return count