def is_palindrome(s):
    return s == s[::-1]

def is_valid_char(char):
    return char.isdigit() and '1' <= char <= '9' or char == '*'

def is_valid_string(s):
    if not s:
        return False
    if not s[0].isdigit():
        return False
    for char in s:
        if not is_valid_char(char):
            return False
    return is_palindrome(s) and len(s) <= 1000

def generate_palindromes(length):
    if length <= 0:
        return
    if length == 1:
        for digit in '123456789':
            yield digit
        return
    if length == 2:
        for digit in '123456789':
            yield digit * 2
        return

    first_chars = '123456789'
    middle_chars = '123456789*'

    for first_char in first_chars:
        middle_lengths = length - 2
        if middle_lengths == 0:
            yield first_char * 2
        elif middle_lengths == 1:
            for mid_char in middle_chars:
                yield first_char + mid_char + first_char
        else:
            for middle_palindrome in generate_palindromes(middle_lengths):
                yield first_char + middle_palindrome + first_char

def solve():
    n = int(input())
    if 1 <= n <= 9:
        return str(n)

    str_n = str(n)
    valid_n_str = True
    for char in str_n:
        if char == '0':
            valid_n_str = False
            break
    if valid_n_str and is_palindrome(str_n):
        val = eval(str_n)
        if val == n:
            return str_n

    for length in range(3, 16):
        for s in generate_palindromes(length):
            if len(s) > 1000:
                continue
            if is_valid_string(s):
                try:
                    if eval(s) == n:
                        return s
                except:
                    pass
    return "-1"

if __name__ == '__main__':
    result = solve()
    print(result)