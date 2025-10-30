from itertools import permutations

def is_palindrome(s):
    return s == s[::-1]

def count_non_palindromic_permutations(N, K, S):
    perm_set = set(permutations(S))
    count = 0
    for perm in perm_set:
        perm_str = ''.join(perm)
        contains_palindrome = False
        for i in range(N - K + 1):
            if is_palindrome(perm_str[i:i + K]):
                contains_palindrome = True
                break
        if not contains_palindrome:
            count += 1
    return count

# Read input from stdin
N, K = map(int, input().split())
S = input().strip()

# Calculate and print the result
print(count_non_palindromic_permutations(N, K, S))