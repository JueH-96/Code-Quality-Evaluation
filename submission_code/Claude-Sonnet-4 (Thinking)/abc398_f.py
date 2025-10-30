s = input().strip()

def is_palindrome(string):
    return string == string[::-1]

n = len(s)

for i in range(n):
    candidate = s + s[:i][::-1]
    if is_palindrome(candidate):
        print(candidate)
        break