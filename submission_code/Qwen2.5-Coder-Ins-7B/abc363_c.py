import sys
from itertools import permutations

def is_palindrome(s, k):
    n = len(s)
    for i in range(n - k + 1):
        if s[i:i+k] == s[i:i+k][::-1]:
            return True
    return False

def count_valid_permutations(n, k, s):
    count = 0
    for perm in permutations(s):
        perm_str = ''.join(perm)
        if not is_palindrome(perm_str, k):
            count += 1
    return count

# Read input
input = sys.stdin.read
data = input().split()
n = int(data[0])
k = int(data[1])
s = data[2]

# Calculate and print the result
result = count_valid_permutations(n, k, s)
print(result)