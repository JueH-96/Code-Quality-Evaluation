# YOUR CODE HERE
s = input()
pos = {}
for i in range(26):
    pos[s[i]] = i + 1
ans = 0
cur = pos['A']
for i in range(ord('A'), ord('Z') + 1):
    c = chr(i)
    ans += abs(pos[c] - cur)
    cur = pos[c]
print(ans)