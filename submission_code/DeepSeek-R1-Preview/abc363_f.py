import math

def is_palindrome(s):
    return s == s[::-1]

def has_zero(s):
    return '0' in s

def find_s(N):
    # Step 1: Check if N itself is a palindrome with no zeros
    s_N = str(N)
    if is_palindrome(s_N) and not has_zero(s_N):
        return s_N
    
    # Step 2: Check two-factor solutions
    max_A = int(math.isqrt(N)) + 1
    for A in range(1, max_A + 1):
        if N % A != 0:
            continue
        B = N // A
        s1 = f"{A}*{B}"
        s2 = f"{B}*{A}"
        if is_palindrome(s1) and not has_zero(s1):
            return s1
        if is_palindrome(s2) and not has_zero(s2):
            return s2
    
    # Step 3: Check three-factor solutions of the form A*B*A
    max_A_three = int(math.isqrt(N)) + 1
    for A in range(1, max_A_three + 1):
        if A * A > N:
            break
        if N % (A * A) != 0:
            continue
        B = N // (A * A)
        s = f"{A}*{B}*{A}"
        if is_palindrome(s) and not has_zero(s):
            return s
    
    # If no solution found
    return -1

N = int(input())
result = find_s(N)
print(result)