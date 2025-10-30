import math

def has_no_zero(x):
    return '0' not in str(x)

def is_palindrome(x):
    s = str(x)
    return s == s[::-1] and has_no_zero(x)

def reverse(x):
    return int(str(x)[::-1]) if x != 0 else 0

def is_perfect_power(n):
    if n == 1:
        return (1, 1)
    max_exp = int(math.log2(n)) + 1
    for k in range(max_exp, 1, -1):
        a = round(n ** (1.0 / k))
        candidates = [a - 1, a, a + 1]
        for candidate in candidates:
            if candidate <= 0:
                continue
            if candidate ** k == n:
                return (candidate, k)
    return None

def find_palindromic_formula(N):
    # Case 1: Check if N itself is a palindrome with no zeros
    if is_palindrome(N) and has_no_zero(N):
        return str(N)
    
    # Case 2: Check if N is a perfect power
    power_result = is_perfect_power(N)
    if power_result:
        a, k = power_result
        if has_no_zero(a):
            return '*'.join([str(a)] * k)
    
    # Case 3-7: Iterate through possible factors
    max_a = int(math.isqrt(N)) + 1
    for a in range(1, max_a + 1):
        if N % a != 0:
            continue
        reversed_a = reverse(a)
        if not has_no_zero(a) or reversed_a == 0 or not has_no_zero(reversed_a):
            continue
        
        b_large = N // a
        # Check two-factor case (a and reversed_a)
        if reversed_a == b_large and has_no_zero(b_large):
            return f"{a}*{reversed_a}"
        
        product_ab = a * reversed_a
        if product_ab == 0:
            continue
        
        # Case 5: Three-factor case with single digit middle
        if N % product_ab == 0:
            c = N // product_ab
            if 1 <= c <= 9 and has_no_zero(c):
                return f"{a}*{c}*{reversed_a}"
        
        # Case 4: Three-factor case with palindrome middle
        if N % (a * a) == 0:
            pal = N // (a * a)
            if is_palindrome(pal) and has_no_zero(pal):
                return f"{a}*{pal}*{a}"
        
        # Case 6: Five-factor case
        quotient = N // product_ab
        max_b = int(math.isqrt(quotient)) + 1
        for b in range(1, max_b + 1):
            if quotient % b == 0:
                reversed_b = reverse(b)
                if not has_no_zero(b) or reversed_b == 0 or not has_no_zero(reversed_b):
                    continue
                product_b = b * reversed_b
                if product_b != 0 and quotient % product_b == 0:
                    c = quotient // product_b
                    if has_no_zero(c):
                        return f"{a}*{b}*{c}*{reversed_b}*{reversed_a}"
                
                # Check the other factor pair (b_large)
                other_b = quotient // b
                reversed_other_b = reverse(other_b)
                if not has_no_zero(other_b) or reversed_other_b == 0 or not has_no_zero(reversed_other_b):
                    continue
                product_other_b = other_b * reversed_other_b
                if product_other_b != 0 and quotient % product_other_b == 0:
                    c_other = quotient // product_other_b
                    if has_no_zero(c_other):
                        return f"{a}*{other_b}*{c_other}*{reversed_other_b}*{reversed_a}"
    return "-1"

N = int(input().strip())
result = find_palindromic_formula(N)
print(result)