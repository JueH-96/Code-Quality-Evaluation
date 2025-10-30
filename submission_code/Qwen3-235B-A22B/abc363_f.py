import math

def has_zero(n):
    return '0' in str(n)

def is_palindrome(s):
    return s == s[::-1]

def solve(N):
    str_n = str(N)
    # Step 1: Check if N itself is a valid palindrome with no zeros
    if not has_zero(N) and is_palindrome(str_n):
        return str_n
    
    max_x = int(math.isqrt(N)) + 1
    
    # Step 2: Check for two-term solution X * rev(X) = N
    for x in range(1, max_x):
        rev_x = int(str(x)[::-1])
        if has_zero(x) or has_zero(rev_x):
            continue
        if x * rev_x == N:
            return f"{x}*{rev_x}"
    
    # Step 3: Check for three-term solution X * Y * rev(X), Y is palindrome
    for x in range(1, max_x):
        rev_x = int(str(x)[::-1])
        if has_zero(x) or has_zero(rev_x):
            continue
        product_x_rev = x * rev_x
        if product_x_rev == 0:
            continue
        if N % product_x_rev != 0:
            continue
        y = N // product_x_rev
        if has_zero(y):
            continue
        s_y = str(y)
        if is_palindrome(s_y):
            return f"{x}*{y}*{rev_x}"
    
    # Additional steps could be added here for more complex cases
    
    # If no solution found
    return -1

if __name__ == "__main__":
    import sys
    N = int(sys.stdin.read())
    result = solve(N)
    print(result)