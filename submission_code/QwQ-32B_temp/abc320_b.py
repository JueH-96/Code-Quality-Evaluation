s = input().strip()

def max_palindrome(s):
    n = len(s)
    max_len = 0

    def expand(l, r):
        while l >= 0 and r < n and s[l] == s[r]:
            l -= 1
            r += 1
        return r - l - 1

    for i in range(n):
        len1 = expand(i, i)
        len2 = expand(i, i + 1)
        max_len = max(max_len, len1, len2)
    return max_len

print(max_palindrome(s))