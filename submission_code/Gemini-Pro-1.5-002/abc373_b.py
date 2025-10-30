# YOUR CODE HERE
s = input()
pos = {}
for i in range(26):
    pos[s[i]] = i + 1

ans = 0
curr = pos['A']
for char_code in range(ord('B'), ord('Z') + 1):
    char = chr(char_code)
    ans += abs(pos[char] - curr)
    curr = pos[char]

print(ans)