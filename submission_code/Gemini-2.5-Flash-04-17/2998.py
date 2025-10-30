class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        
        def is_symmetric(n: int) -> bool:
            """Checks if a number is symmetric based on the definition."""
            s = str(n)
            n_str = len(s)
            
            # Must have an even number of digits
            if n_str % 2 != 0:
                return False
            
            half_len = n_str // 2
            
            # Calculate sum of first half digits
            sum_first = sum(int(d) for d in s[:half_len])
                
            # Calculate sum of second half digits
            sum_second = sum(int(d) for d in s[half_len:])
            
            # Check if sums are equal
            return sum_first == sum_second

        count = 0
        # Iterate through the range [low, high] inclusive
        for num in range(low, high + 1):
            if is_symmetric(num):
                count += 1
                
        return count