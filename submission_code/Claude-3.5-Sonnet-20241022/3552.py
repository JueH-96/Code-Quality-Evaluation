class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        # For single digit case
        if n == 1:
            # Try largest possible single digit divisible by k
            for i in range(9, 0, -1):
                if i % k == 0:
                    return str(i)
            
        # For multi-digit case
        # Start with largest possible n-digit number
        start = 10**n - 1
        
        # Try each number from largest to smallest
        for i in range(start, 10**(n-1)-1, -1):
            # Generate palindrome by mirroring left half
            left_half = str(i)
            if n % 2 == 0:
                # For even length, mirror all digits
                num = int(left_half + left_half[::-1])
            else:
                # For odd length, exclude middle digit when mirroring
                num = int(left_half + left_half[:-1][::-1])
                
            # Check if number is divisible by k
            if num % k == 0:
                return str(num)