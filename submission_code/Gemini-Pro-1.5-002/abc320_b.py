# YOUR CODE HERE
def is_palindrome(s):
    return s == s[::-1]

s = input()
n = len(s)
max_len = 0

for i in range(n):
    for j in range(i, n):
        substring = s[i:j+1]
        if is_palindrome(substring):
            max_len = max(max_len, len(substring))

print(max_len)