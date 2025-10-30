N = int(input())

def is_palindrome(s):
    return s == s[::-1]

def has_zero(n):
    return '0' in str(n)

def reverse_number(n):
    """Reverse the digits of a number"""
    return int(str(n)[::-1])

# Case 1: N itself is a palindrome without zeros
if is_palindrome(str(N)) and not has_zero(N):
    print(str(N))
else:
    found = False
    
    # Case 2: Pattern a*b*a where a^2 * b = N
    if not found:
        for a in range(1, min(1000000, int(N**0.5) + 1)):
            if has_zero(a) or N % (a * a) != 0:
                continue
            b = N // (a * a)
            if has_zero(b):
                continue
            expr = f"{a}*{b}*{a}"
            if is_palindrome(expr) and len(expr) <= 1000:
                print(expr)
                found = True
                break
    
    # Case 3: Pattern a*b*c*reverse(b)*a where c is palindrome
    if not found:
        for a in range(1, 100):
            if has_zero(a) or N % (a * a) != 0:
                continue
            remaining = N // (a * a)
            
            for b in range(1, 10000):
                if has_zero(b):
                    continue
                b_rev = reverse_number(b)
                if has_zero(b_rev) or b_rev == 0:
                    continue
                
                if remaining % (b * b_rev) != 0:
                    continue
                c = remaining // (b * b_rev)
                
                if has_zero(c):
                    continue
                
                if is_palindrome(str(c)):
                    expr = f"{a}*{b}*{c}*{b_rev}*{a}"
                    if is_palindrome(expr) and len(expr) <= 1000:
                        print(expr)
                        found = True
                        break
            
            if found:
                break
    
    # Case 4: Try more general factorizations
    if not found:
        factors = []
        for i in range(1, min(100000, int(N**0.5) + 1)):
            if N % i == 0 and not has_zero(i):
                factors.append(i)
                if i != N // i and not has_zero(N // i):
                    factors.append(N // i)
        
        # Try 3-factor combinations
        for i in range(len(factors)):
            for j in range(len(factors)):
                if N % (factors[i] * factors[j]) != 0:
                    continue
                k = N // (factors[i] * factors[j])
                if not has_zero(k):
                    expr = f"{factors[i]}*{factors[j]}*{k}"
                    if is_palindrome(expr) and len(expr) <= 1000:
                        print(expr)
                        found = True
                        break
            if found:
                break
    
    if not found:
        print("-1")