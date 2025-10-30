def shortest_palindrome(s):
    # If s is already a palindrome, return s
    if s == s[::-1]:
        return s
    
    # Find the largest k such that s[len(s)-k:] is a palindrome
    k = 0
    for i in range(len(s), 0, -1):
        suffix = s[len(s)-i:]
        if suffix == suffix[::-1]:
            k = i
            break
    
    # Append the reverse of the non-palindromic prefix to s
    return s + s[:len(s)-k][::-1]

# Read input
s = input().strip()

# Print output
print(shortest_palindrome(s))