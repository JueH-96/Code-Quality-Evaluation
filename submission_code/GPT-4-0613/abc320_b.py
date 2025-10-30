def is_palindrome(s):
    return s == s[::-1]

def max_palindrome_length(s):
    for length in range(len(s), 0, -1):
        for start in range(len(s) - length + 1):
            if is_palindrome(s[start:start+length]):
                return length

s = input().strip()
print(max_palindrome_length(s))