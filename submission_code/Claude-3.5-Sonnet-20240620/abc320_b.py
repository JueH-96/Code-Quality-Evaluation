# YOUR CODE HERE
def longest_palindrome_substring(s):
    n = len(s)
    max_length = 1
    
    # Function to expand around center
    def expand_around_center(left, right):
        while left >= 0 and right < n and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1
    
    for i in range(n):
        # Odd length palindromes
        length1 = expand_around_center(i, i)
        # Even length palindromes
        length2 = expand_around_center(i, i + 1)
        
        max_length = max(max_length, length1, length2)
    
    return max_length

# Read input
S = input().strip()

# Solve and print output
print(longest_palindrome_substring(S))