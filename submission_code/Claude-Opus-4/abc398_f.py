# YOUR CODE HERE
S = input().strip()
n = len(S)

# Check if the entire string is already a palindrome
def is_palindrome(s):
    return s == s[::-1]

# Find the longest prefix of S that is a palindrome
for i in range(n, 0, -1):
    if is_palindrome(S[:i]):
        # We found the longest palindrome prefix
        # Append the reverse of the remaining part
        result = S + S[i:][::-1]
        print(result)
        break