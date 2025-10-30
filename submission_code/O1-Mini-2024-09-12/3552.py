class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        # Initialize the palindrome as a list of '9's
        palindrome = ['9'] * n
        
        # Function to compute modulo
        def compute_mod(s, k):
            mod = 0
            for c in s:
                mod = (mod * 10 + int(c)) % k
            return mod
        
        mod = compute_mod(palindrome, k)
        
        if mod == 0:
            return ''.join(palindrome)
        
        # If mod != 0, need to subtract mod to make it divisible by k
        # Implement subtraction with palindrome constraints
        # Start from the end
        carry = mod
        for i in range(n-1, -1, -1):
            digit = int(palindrome[i]) - carry
            if digit < 0:
                digit += 10
                carry = 1
            else:
                carry = 0
            palindrome[i] = str(digit)
            # Mirror the digit
            if i < n // 2:
                palindrome[n - 1 - i] = palindrome[i]
            if carry == 0:
                break
        # After adjustment, check if first digit is not zero
        if palindrome[0] == '0':
            return "-1"  # No such palindrome exists
        # Now, recompute mod
        mod = compute_mod(palindrome, k)
        if mod == 0:
            return ''.join(palindrome)
        else:
            return "-1"