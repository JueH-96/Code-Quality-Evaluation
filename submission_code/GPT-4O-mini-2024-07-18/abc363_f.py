def is_palindrome(s):
    return s == s[::-1]

def generate_palindrome_expression(n):
    # We will try to find a valid expression that evaluates to n and is a palindrome
    # We will use a simple approach to generate factors of n
    factors = []
    
    # Finding factors of n
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            factors.append(i)
            if i != n // i:
                factors.append(n // i)
    
    # We will try to create a palindrome from the factors
    for factor in factors:
        # Convert factor to string
        factor_str = str(factor)
        if '0' in factor_str:
            continue  # Skip factors containing '0'
        
        # Create a potential palindrome
        palindrome_candidate = factor_str + '*' + factor_str[::-1]
        
        # Evaluate the palindrome candidate
        try:
            if eval(palindrome_candidate) == n and is_palindrome(palindrome_candidate):
                return palindrome_candidate
        except:
            continue
    
    # If no valid palindrome expression found, return -1
    return -1

import sys
input = sys.stdin.read

N = int(input().strip())
result = generate_palindrome_expression(N)
print(result)