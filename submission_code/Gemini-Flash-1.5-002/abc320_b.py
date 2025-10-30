# YOUR CODE HERE
s = input()
n = len(s)
ans = 1
for i in range(n):
    l, r = i, i
    while l >= 0 and r < n and s[l] == s[r]:
        ans = max(ans, r - l + 1)
        l -= 1
        r += 1
    l, r = i, i + 1
    while l >= 0 and r < n and s[l] == s[r]:
        ans = max(ans, r - l + 1)
        l -= 1
        r += 1
print(ans)