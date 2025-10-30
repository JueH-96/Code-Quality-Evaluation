class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        # Special case for n = 1
        if n == 1:
            for digit in range(9, 0, -1):
                if digit % k == 0:
                    return str(digit)
        
        # For n > 1, we need to construct palindromes
        # Try to construct the largest palindrome by setting digits
        
        # Start with all 9s
        digits = ['9'] * n
        
        # For a palindrome, digits[i] must equal digits[n-1-i]
        # We only need to work with the first half (and middle if odd)
        half = (n + 1) // 2
        
        # Try all possible combinations starting from the largest
        def generate_palindrome(half_digits):
            palindrome = half_digits[:]
            if n % 2 == 1:
                # Odd length - don't duplicate the middle digit
                palindrome.extend(half_digits[-2::-1])
            else:
                # Even length - mirror the entire first half
                palindrome.extend(half_digits[::-1])
            return palindrome
        
        # Try all possible values for the first half
        def try_palindromes(pos, current):
            if pos == half:
                # Generate the full palindrome
                palindrome = generate_palindrome(current)
                num_str = ''.join(palindrome)
                num = int(num_str)
                if num % k == 0:
                    return num_str
                return None
            
            # Try digits from 9 to 0 for this position
            start = 9 if pos > 0 else 9  # First digit can't be 0
            end = 0 if pos > 0 else 1
            
            for digit in range(start, end - 1, -1):
                current[pos] = str(digit)
                result = try_palindromes(pos + 1, current)
                if result:
                    return result
            
            return None
        
        # Start with the first half
        current = ['9'] * half
        result = try_palindromes(0, current)
        
        return result