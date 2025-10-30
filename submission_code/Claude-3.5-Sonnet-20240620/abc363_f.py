# YOUR CODE HERE
def is_palindrome(s):
    return s == s[::-1]

def evaluate(s):
    return eval(s.replace('*', '*1'))

def generate_palindrome(n):
    for length in range(1, 1001):
        for first_digit in range(1, 10):
            if length == 1:
                if first_digit == n:
                    return str(first_digit)
            else:
                mid = (length + 1) // 2
                for i in range(10**(mid-1), 10**mid):
                    left = str(first_digit) + str(i)[1:]
                    if length % 2 == 0:
                        s = left + left[::-1]
                    else:
                        s = left + left[-2::-1]
                    
                    s_with_mult = ''
                    for c in s:
                        if c != '0':
                            s_with_mult += c
                            s_with_mult += '*'
                    s_with_mult = s_with_mult[:-1]
                    
                    if is_palindrome(s_with_mult) and evaluate(s_with_mult) == n:
                        return s_with_mult
    return '-1'

n = int(input())
print(generate_palindrome(n))