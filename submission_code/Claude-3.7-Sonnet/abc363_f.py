def is_palindrome(s):
    return s == s[::-1]

def contains_zero(s):
    return '0' in s

def check_a_a(N):
    """Check if N can be represented as a*a."""
    import math
    a = int(math.sqrt(N))
    if a * a == N and not contains_zero(str(a)):
        formula = str(a) + "*" + str(a)
        if len(formula) <= 1000:
            return formula
    return None

def check_a_b_a(N):
    """Check if N can be represented as a*b*a."""
    import math
    a_limit = min(10000, int(math.sqrt(N)))
    
    for a in range(1, a_limit + 1):
        if contains_zero(str(a)):
            continue
        
        a_squared = a * a
        if N % a_squared == 0:
            b = N // a_squared
            
            if not contains_zero(str(b)):
                formula = str(a) + "*" + str(b) + "*" + str(a)
                if len(formula) <= 1000:
                    return formula
    
    return None

def check_a_b_c_b_a(N):
    """Check if N can be represented as a*b*c*b*a."""
    import math
    a_limit = min(1000, int(math.sqrt(N)))
    
    for a in range(1, a_limit + 1):
        if contains_zero(str(a)):
            continue
        
        a_squared = a * a
        b_limit = min(1000, int(math.sqrt(N / a_squared)))
        
        for b in range(1, b_limit + 1):
            if contains_zero(str(b)):
                continue
            
            b_squared = b * b
            if N % (a_squared * b_squared) == 0:
                c = N // (a_squared * b_squared)
                
                if not contains_zero(str(c)):
                    formula = str(a) + "*" + str(b) + "*" + str(c) + "*" + str(b) + "*" + str(a)
                    if len(formula) <= 1000:
                        return formula
    
    return None

def check_a_b_c_d_c_b_a(N):
    """Check if N can be represented as a*b*c*d*c*b*a."""
    import math
    a_limit = min(100, int(math.sqrt(N)))
    
    for a in range(1, a_limit + 1):
        if contains_zero(str(a)):
            continue
        
        a_squared = a * a
        b_limit = min(100, int(math.sqrt(N / a_squared)))
        
        for b in range(1, b_limit + 1):
            if contains_zero(str(b)):
                continue
            
            b_squared = b * b
            c_limit = min(100, int(math.sqrt(N / (a_squared * b_squared))))
            
            for c in range(1, c_limit + 1):
                if contains_zero(str(c)):
                    continue
                
                c_squared = c * c
                if N % (a_squared * b_squared * c_squared) == 0:
                    d = N // (a_squared * b_squared * c_squared)
                    
                    if not contains_zero(str(d)):
                        formula = str(a) + "*" + str(b) + "*" + str(c) + "*" + str(d) + "*" + str(c) + "*" + str(b) + "*" + str(a)
                        if len(formula) <= 1000:
                            return formula
    
    return None

def solve(N):
    """Find a palindromic formula for N."""
    N_str = str(N)
    
    # Check if N itself is a palindrome and doesn't contain 0
    if is_palindrome(N_str) and not contains_zero(N_str):
        return N_str
    
    # Check pattern a*a
    result = check_a_a(N)
    if result:
        return result
    
    # Check pattern a*b*a
    result = check_a_b_a(N)
    if result:
        return result
    
    # Check pattern a*b*c*b*a
    result = check_a_b_c_b_a(N)
    if result:
        return result
    
    # Check pattern a*b*c*d*c*b*a
    result = check_a_b_c_d_c_b_a(N)
    if result:
        return result
    
    return "-1"  # Default return if no solution is found

if __name__ == "__main__":
    N = int(input().strip())
    print(solve(N))