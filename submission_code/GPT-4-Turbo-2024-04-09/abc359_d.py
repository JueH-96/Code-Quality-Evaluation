def is_palindrome(s):
    return s == s[::-1]

def count_good_strings(N, K, S):
    MOD = 998244353
    # Count the number of '?' in S
    q = S.count('?')
    
    # If there are no '?' characters, we only need to check the given string
    if q == 0:
        for i in range(N - K + 1):
            if is_palindrome(S[i:i+K]):
                return 0
        return 1
    
    # Generate all possible combinations for the '?' characters
    from itertools import product
    
    # Replace '?' with indices in the string
    indices = [i for i, char in enumerate(S) if char == '?']
    total_good_strings = 0
    
    # Iterate over all possible replacements of '?' with 'A' or 'B'
    for replacement in product('AB', repeat=q):
        # Create a list from the string to modify it
        new_string = list(S)
        for idx, char in zip(indices, replacement):
            new_string[idx] = char
        
        # Join to form the new candidate string
        candidate = ''.join(new_string)
        
        # Check if this candidate string is good
        is_good = True
        for i in range(N - K + 1):
            if is_palindrome(candidate[i:i+K]):
                is_good = False
                break
        
        if is_good:
            total_good_strings += 1
    
    return total_good_strings % MOD

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
K = int(data[1])
S = data[2]

print(count_good_strings(N, K, S))