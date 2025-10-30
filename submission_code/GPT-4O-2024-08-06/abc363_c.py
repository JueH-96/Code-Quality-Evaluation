from itertools import permutations

def is_palindrome(s):
    return s == s[::-1]

def count_non_palindromic_permutations(N, K, S):
    unique_permutations = set(permutations(S))
    count = 0
    
    for perm in unique_permutations:
        perm_str = ''.join(perm)
        has_palindrome = False
        
        for i in range(N - K + 1):
            if is_palindrome(perm_str[i:i+K]):
                has_palindrome = True
                break
        
        if not has_palindrome:
            count += 1
    
    return count

# Read input
import sys
input = sys.stdin.read
data = input().strip().split()
N = int(data[0])
K = int(data[1])
S = data[2]

# Calculate and print the result
result = count_non_palindromic_permutations(N, K, S)
print(result)