class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        """
        Counts the number of symmetric integers in the range [low, high].

        An integer is symmetric if it has an even number of digits and the sum
        of the first half of the digits is equal to the sum of the second half.
        """
        count = 0
        
        # Iterate through each number in the inclusive range [low, high].
        for num in range(low, high + 1):
            s = str(num)
            n = len(s)
            
            # A number can only be symmetric if it has an even number of digits.
            if n % 2 == 0:
                half = n // 2
                
                # Calculate the sum of digits for the first and second halves.
                sum_first_half = sum(int(digit) for digit in s[:half])
                sum_second_half = sum(int(digit) for digit in s[half:])
                
                # If the sums are equal, the number is symmetric.
                if sum_first_half == sum_second_half:
                    count += 1
                    
        return count