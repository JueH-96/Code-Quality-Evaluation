from itertools import permutations

def contains_palindrome_of_length_k(s, k):
    n = len(s)
    # Check all substrings of length k
    for i in range(n - k + 1):
        substring = s[i:i+k]
        # Check if substring is a palindrome
        if substring == substring[::-1]:
            return True
    return False

# Read input
n, k = map(int, input().split())
s = input().strip()

# Generate all unique permutations
unique_perms = set(permutations(s))

# Count permutations that don't contain a palindrome of length k
count = 0
for perm in unique_perms:
    perm_str = ''.join(perm)
    if not contains_palindrome_of_length_k(perm_str, k):
        count += 1

print(count)