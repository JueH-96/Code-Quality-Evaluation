s = input()

ans = 0
for i in range(len(s)):
    # even palindromes
    l, r = i, i + 1
    while l >= 0 and r < len(s) and s[l] == s[r]:
        ans = max(ans, r - l + 1)
        l -= 1
        r += 1

    # odd palindromes
    l, r = i, i
    while l >= 0 and r < len(s) and s[l] == s[r]:
        ans = max(ans, r - l + 1)
        l -= 1
        r += 1

print(ans)