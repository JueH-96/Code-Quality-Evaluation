from itertools import permutations

def has_palindrome_of_length_k(s, k):
    n = len(s)
    for i in range(n - k + 1):
        if s[i:i+k] == s[i:i+k][::-1]:
            return True
    return False

def count_valid_permutations(s, k):
    count = 0
    for perm in permutations(s):
        perm_str = ''.join(perm)
        if not has_palindrome_of_length_k(perm_str, k):
            count += 1
    return count

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
S = data[2]

# Calculate and print the result
result = count_valid_permutations(S, K)
print(result)