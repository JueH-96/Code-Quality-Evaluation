from itertools import permutations

def has_palindrome_substring(s, k):
    n = len(s)
    # Check all possible substrings of length k
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
all_perms = set([''.join(p) for p in permutations(s)])

# Count permutations that do NOT contain palindrome of length k
count = 0
for perm in all_perms:
    if not has_palindrome_substring(perm, k):
        count += 1

print(count)