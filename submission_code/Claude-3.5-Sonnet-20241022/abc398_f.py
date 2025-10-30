def shortest_palindrome(s):
    # If string is already a palindrome, return it
    if s == s[::-1]:
        return s
    
    # Find the longest palindrome prefix
    n = len(s)
    for i in range(n, -1, -1):
        if s[:i] == s[:i][::-1]:
            # Once found, append the reverse of remaining characters
            return s + s[i:][::-1]

# Read input
S = input().strip()

# Get and print result
result = shortest_palindrome(S)
print(result)