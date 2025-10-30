import sys
import itertools

def has_palindrome_substring(s, K):
    N = len(s)
    for i in range(N - K + 1):
        if is_palindrome(s, i, K):
            return True
    return False

def is_palindrome(s, start, K):
    for d in range(K // 2):
        if s[start + d] != s[start + K - 1 - d]:
            return False
    return True

# Read input
data = sys.stdin.read().split()
N = int(data[0])
K = int(data[1])
S = data[2]

# Generate all unique permutations as strings
perm_strings = set(''.join(p) for p in itertools.permutations(S))

# Count how many do not have palindrome substring of length K
count = 0
for perm in perm_strings:
    if not has_palindrome_substring(perm, K):
        count += 1

# Output the count
print(count)