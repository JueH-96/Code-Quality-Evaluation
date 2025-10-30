class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        
        if n < 4:
            return 0
        
        # Count of ways to arrange "leet" in a string of length n
        # We need to place "leet" and fill the rest with any characters
        # The number of ways to arrange "leet" is fixed (4! / (2! * 2!))
        leet_count = 6  # 4! / (2! * 2!) = 24 / 4 = 6
        
        # Remaining characters to fill
        remaining_length = n - 4
        
        # Total characters available (26 lowercase letters)
        total_chars = 26
        
        # Calculate the number of ways to fill the remaining_length with any characters
        # Each position can be filled with any of the 26 characters
        # So we have total_chars^remaining_length ways to fill the rest
        ways_to_fill = pow(total_chars, remaining_length, MOD)
        
        # Total good strings
        total_good_strings = (leet_count * ways_to_fill) % MOD
        
        return total_good_strings