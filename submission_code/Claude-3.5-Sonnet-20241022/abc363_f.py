def is_palindrome(s):
    return s == s[::-1]

def evaluate(s):
    try:
        return eval(s)
    except:
        return None

def solve(N):
    # Try single digit first
    if N <= 9:
        return str(N)
    
    # Try palindromes without multiplication
    s = str(N)
    if is_palindrome(s) and '0' not in s:
        return s

    # Try palindromes with multiplication
    # Length of palindrome will be odd or even
    for length in range(3, 21, 2):  # Odd lengths
        # Generate first half
        half_len = length // 2
        for first_digit in range(1, 10):
            first_half = [str(first_digit)]
            
            def generate(curr_half):
                if len(curr_half) == half_len:
                    # Complete palindrome
                    mid = ['*', str(first_digit), '*'] # Try middle digit
                    full = curr_half + mid + curr_half[::-1]
                    s = ''.join(full)
                    val = evaluate(s)
                    if val == N:
                        return s
                    
                    # Try without middle multiplication
                    for d in range(1, 10):
                        full = curr_half + [str(d)] + curr_half[::-1]
                        s = ''.join(full)
                        val = evaluate(s)
                        if val == N:
                            return s
                    return None
                
                for d in range(1, 10):
                    curr_half.append(str(d))
                    res = generate(curr_half)
                    if res:
                        return res
                    curr_half.pop()
                
                curr_half.append('*')
                res = generate(curr_half)
                if res:
                    return res
                curr_half.pop()
                
                return None
            
            result = generate(first_half)
            if result:
                return result

    # Try even length palindromes
    for length in range(2, 20, 2):
        for first_digit in range(1, 10):
            first_half = [str(first_digit)]
            
            def generate(curr_half):
                if len(curr_half) == length//2:
                    full = curr_half + curr_half[::-1]
                    s = ''.join(full)
                    val = evaluate(s)
                    if val == N:
                        return s
                    return None
                
                for d in range(1, 10):
                    curr_half.append(str(d))
                    res = generate(curr_half)
                    if res:
                        return res
                    curr_half.pop()
                
                curr_half.append('*')
                res = generate(curr_half)
                if res:
                    return res
                curr_half.pop()
                
                return None
            
            result = generate(first_half)
            if result:
                return result
                
    return "-1"

N = int(input())
print(solve(N))