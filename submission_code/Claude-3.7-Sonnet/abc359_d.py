def count_good_strings(s, k):
    n = len(s)
    MOD = 998244353
    memo = {}
    filled = [''] * n  # Track characters we've filled
    
    def is_palindrome(start, end):
        """Check if substring from start to end is a palindrome."""
        while start < end:
            if filled[start] != filled[end]:
                return False
            start += 1
            end -= 1
        return True
    
    def dp(pos):
        """
        Calculate the number of good strings that can be formed 
        starting from position pos.
        """
        if pos == n:
            return 1  # Successfully filled the entire string
        
        # Only need to remember characters in the last k-1 positions for memoization
        state = (pos, tuple(filled[max(0, pos-k+1):pos]))
        if state in memo:
            return memo[state]
        
        count = 0
        choices = ['A', 'B'] if s[pos] == '?' else [s[pos]]
        
        for char in choices:
            filled[pos] = char
            
            # Check if placing this character creates a palindrome
            creates_palindrome = False
            for start in range(0, max(0, pos - k + 1) + 1):
                end = start + k - 1
                if end < n and is_palindrome(start, end):
                    creates_palindrome = True
                    break
            
            if not creates_palindrome:
                count = (count + dp(pos + 1)) % MOD
            
            # Backtrack
            filled[pos] = ''
        
        memo[state] = count
        return count
    
    return dp(0)

# Read input
n, k = map(int, input().split())
s = input().strip()

# Compute and output the result
result = count_good_strings(s, k)
print(result)