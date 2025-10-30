from itertools import permutations

def is_palindrome(s):
    """Check if a string is a palindrome."""
    return s == s[::-1]

def has_palindrome_substring(s, k):
    """Check if string s has a palindrome substring of length k."""
    for i in range(len(s) - k + 1):
        if is_palindrome(s[i:i+k]):
            return True
    return False

def count_permutations_without_palindrome(s, k):
    """Count permutations of s that don't contain a palindrome of length k."""
    valid_count = 0
    
    for p in permutations(s):
        p_str = ''.join(p)
        if not has_palindrome_substring(p_str, k):
            valid_count += 1
    
    return valid_count

# Read input
n, k = map(int, input().split())
s = input().strip()

# Calculate and output the result
print(count_permutations_without_palindrome(s, k))