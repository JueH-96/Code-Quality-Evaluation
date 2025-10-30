from itertools import permutations

def has_palindrome_substring(s, k):
    """Check if string s contains a palindrome of length k as a substring"""
    n = len(s)
    for i in range(n - k + 1):
        # Check if substring s[i:i+k] is a palindrome
        is_palindrome = True
        for j in range(k // 2):
            if s[i + j] != s[i + k - 1 - j]:
                is_palindrome = False
                break
        if is_palindrome:
            return True
    return False

# Read input
n, k = map(int, input().split())
s = input().strip()

# Generate all unique permutations
unique_perms = set(permutations(s))

# Count permutations that don't contain palindrome of length k
count = 0
for perm in unique_perms:
    perm_str = ''.join(perm)
    if not has_palindrome_substring(perm_str, k):
        count += 1

print(count)