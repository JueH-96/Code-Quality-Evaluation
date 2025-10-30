class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def is_beautiful(n: int) -> bool:
            """Check if a number is beautiful."""
            # Convert the number to a string to easily count even and odd digits
            str_n = str(n)
            even_count = 0
            odd_count = 0
            
            # Count even and odd digits
            for digit in str_n:
                if int(digit) % 2 == 0:
                    even_count += 1
                else:
                    odd_count += 1
            
            # Check if the number is beautiful
            return even_count == odd_count and n % k == 0

        # Initialize the count of beautiful integers
        beautiful_count = 0
        
        # Iterate over the range [low, high]
        for num in range(low, high + 1):
            # Check if the current number is beautiful
            if is_beautiful(num):
                beautiful_count += 1
        
        return beautiful_count