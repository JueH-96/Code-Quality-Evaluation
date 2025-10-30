def expand(s, l, r):
    while l >= 0 and r < len(s) and s[l] == s[r]:
        l -= 1
        r += 1
    return r - l - 1

s = input().strip()
max_len = 1
for i in range(len(s)):
    len1 = expand(s, i, i)
    len2 = expand(s, i, i + 1)
    current_max = max(len1, len2)
    if current_max > max_len:
        max_len = current_max
print(max_len)