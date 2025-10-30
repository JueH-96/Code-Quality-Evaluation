def is_palindrome(s):
    return s == s[::-1]

s = input()
n = len(s)
ans = 1
for i in range(n):
    for j in range(i, n):
        sub = s[i:j+1]
        if is_palindrome(sub):
            ans = max(ans, len(sub))
print(ans)