def is_palindrome(s):
    return s == s[::-1]

def count_good_strings(N, K, S):
    MOD = 998244353
    q = S.count('?')
    total_good_strings = 0
    
    # Generate all combinations of A and B for the '?' characters
    from itertools import product
    
    for replacement in product('AB', repeat=q):
        current_string = list(S)
        replacement_index = 0
        
        # Replace '?' with the current combination
        for i in range(N):
            if current_string[i] == '?':
                current_string[i] = replacement[replacement_index]
                replacement_index += 1
        
        # Check if the resulting string has any palindromic substring of length K
        has_palindrome = False
        for i in range(N - K + 1):
            if is_palindrome(current_string[i:i + K]):
                has_palindrome = True
                break
        
        if not has_palindrome:
            total_good_strings += 1
            total_good_strings %= MOD
    
    return total_good_strings

import sys
input = sys.stdin.read
data = input().splitlines()
N, K = map(int, data[0].split())
S = data[1]

result = count_good_strings(N, K, S)
print(result)