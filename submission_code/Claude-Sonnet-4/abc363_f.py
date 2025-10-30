def is_palindrome(s):
    return s == s[::-1]

def contains_only_valid_chars(s):
    valid_chars = set('123456789*')
    return all(c in valid_chars for c in s)

def evaluate_expression(s):
    try:
        # Replace * with actual multiplication and evaluate
        return eval(s.replace('*', '*'))
    except:
        return None

def get_factors(n):
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def create_palindromic_expression(factors):
    if not factors:
        return None
    
    # Convert factors to strings and filter out those containing 0
    factor_strs = []
    for f in factors:
        f_str = str(f)
        if '0' not in f_str:
            factor_strs.append(f_str)
        else:
            # If any factor contains 0, we can't use it
            return None
    
    if not factor_strs:
        return None
    
    # Sort factors to try different arrangements
    factor_strs.sort()
    
    # Try to create palindrome
    # For even number of factors
    if len(factor_strs) % 2 == 0:
        half = len(factor_strs) // 2
        left_half = factor_strs[:half]
        right_half = factor_strs[half:][::-1]
        expression = '*'.join(left_half + right_half)
    else:
        # For odd number of factors
        half = len(factor_strs) // 2
        left_half = factor_strs[:half]
        middle = factor_strs[half]
        right_half = factor_strs[half+1:][::-1]
        expression = '*'.join(left_half + [middle] + right_half)
    
    return expression

def solve(N):
    # First check if N itself is valid
    n_str = str(N)
    if is_palindrome(n_str) and contains_only_valid_chars(n_str):
        return n_str
    
    # Try to factorize and create palindromic expression
    factors = get_factors(N)
    
    # Try different arrangements of factors
    from itertools import permutations
    
    # For small number of factors, try all permutations
    if len(factors) <= 8:  # Limit to avoid timeout
        for perm in permutations(factors):
            # Try creating palindrome with this permutation
            perm_strs = [str(f) for f in perm if '0' not in str(f)]
            if len(perm_strs) != len(perm):
                continue
                
            # Create palindromic arrangement
            n = len(perm_strs)
            if n == 1:
                expr = perm_strs[0]
            else:
                # Try different palindromic arrangements
                for i in range((n + 1) // 2):
                    left = perm_strs[:i+1]
                    if n % 2 == 1 and i == n // 2:
                        # Odd length, middle element
                        expr = '*'.join(left + left[-2::-1])
                    else:
                        # Even length or not middle
                        right = perm_strs[n-i-1:n-i] if i < n-1 else []
                        remaining = perm_strs[i+1:n-i-1]
                        if remaining:
                            middle_part = '*'.join(remaining + remaining[::-1])
                            expr = '*'.join(left + [middle_part] + right[::-1])
                        else:
                            expr = '*'.join(left + right[::-1])
            
            if (is_palindrome(expr) and 
                contains_only_valid_chars(expr) and 
                expr[0].isdigit() and
                len(expr) <= 1000):
                try:
                    if eval(expr) == N:
                        return expr
                except:
                    continue
    
    # Simple approach: try to create a*b*a format where a*a*b = N
    for a in range(1, 10):
        if N % (a * a) == 0:
            b = N // (a * a)
            b_str = str(b)
            if '0' not in b_str and len(b_str) <= 996:  # Leave room for a*...*a
                expr = f"{a}*{b_str}*{a}"
                if (is_palindrome(expr) and 
                    contains_only_valid_chars(expr) and
                    len(expr) <= 1000):
                    return expr
    
    return "-1"

# Read input
N = int(input())
print(solve(N))