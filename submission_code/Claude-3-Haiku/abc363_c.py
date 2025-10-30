from itertools import permutations

def is_palindrome(s, k):
    for i in range(len(s) - k + 1):
        if s[i:i+k] == s[i+k-1:i-1:-1]:
            return True
    return False

def count_non_palindrome_permutations(s, k):
    n = len(s)
    count = 0
    for perm in permutations(s):
        perm_str = ''.join(perm)
        if not is_palindrome(perm_str, k):
            count += 1
    return count

# Read input
n, k = map(int, input().split())
s = input()

# Solve the problem
result = count_non_palindrome_permutations(s, k)
print(result)