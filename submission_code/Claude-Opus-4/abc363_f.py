# YOUR CODE HERE
def is_palindrome(s):
    return s == s[::-1]

def contains_only_valid_digits(n):
    return all(c in '123456789' for c in str(n))

def factorize(n):
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

def solve(N):
    # Check if N itself is a valid palindrome
    str_n = str(N)
    if is_palindrome(str_n) and contains_only_valid_digits(N):
        return str_n
    
    # Try to find palindromic factorization
    factors = factorize(N)
    
    # Try different combinations of factors
    from itertools import combinations
    
    # Try to group factors to create palindromic expressions
    for r in range(1, min(len(factors) + 1, 10)):  # Limit to avoid too long strings
        for indices in combinations(range(len(factors)), r):
            left_factors = []
            remaining = N
            
            # Calculate left side factors
            for i in indices:
                left_factors.append(factors[i])
                remaining //= factors[i]
            
            # Check if remaining can form the middle and right side
            if remaining == 1:
                # No middle factor needed
                expr_parts = [str(f) for f in left_factors]
                expr = '*'.join(expr_parts + expr_parts[::-1])
                if len(expr) <= 1000 and contains_only_valid_digits(''.join(expr_parts)):
                    return expr
            else:
                # Check if remaining is a valid middle factor
                if contains_only_valid_digits(remaining):
                    expr_parts = [str(f) for f in left_factors]
                    expr = '*'.join(expr_parts + [str(remaining)] + expr_parts[::-1])
                    if len(expr) <= 1000:
                        return expr
    
    # Try a different approach - find two identical factors that multiply to give N
    for i in range(2, int(N**0.5) + 1):
        if N % i == 0:
            quotient = N // i
            if quotient % i == 0 and contains_only_valid_digits(i):
                middle = quotient // i
                if contains_only_valid_digits(middle):
                    expr = f"{i}*{middle}*{i}"
                    if len(expr) <= 1000:
                        return expr
    
    return "-1"

N = int(input())
print(solve(N))