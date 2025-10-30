def solve(N):
    # Case 1: Single digit
    if 1 <= N <= 9:
        return str(N)
    
    # Case 2: N itself is a palindrome with valid digits (no 0)
    n_str = str(N)
    if n_str == n_str[::-1] and all(c in '123456789' for c in n_str):
        return n_str
    
    # Case 3: Pattern a*b*a (single digit a)
    for a in range(1, 10):
        if N % (a * a) == 0:
            b = N // (a * a)
            if b > 0:
                b_str = str(b)
                if all(c in '123456789' for c in b_str):
                    result = f"{a}*{b_str}*{a}"
                    if len(result) <= 1000 and result == result[::-1]:
                        return result
    
    # Case 4: Pattern aa*b*aa (two-digit palindromes)
    for digit in range(1, 10):
        a_str = str(digit) + str(digit)
        a = int(a_str)
        if N % (a * a) == 0:
            b = N // (a * a)
            if b > 0:
                b_str = str(b)
                if all(c in '123456789' for c in b_str):
                    result = f"{a_str}*{b_str}*{a_str}"
                    if len(result) <= 1000 and result == result[::-1]:
                        return result
    
    # Case 5: Pattern a*b*palindrome*reverse(b)*a
    for a in range(1, 10):
        for b in range(12, 100):
            b_str = str(b)
            if '0' in b_str:
                continue
            b_rev_str = b_str[::-1]
            b_rev = int(b_rev_str)
            divisor = a * a * b * b_rev
            if N % divisor == 0:
                c = N // divisor
                if c > 0:
                    c_str = str(c)
                    if c_str == c_str[::-1] and all(c_char in '123456789' for c_char in c_str):
                        result = f"{a}*{b_str}*{c_str}*{b_rev_str}*{a}"
                        if len(result) <= 1000 and result == result[::-1]:
                            return result
    
    return "-1"

N = int(input())
print(solve(N))