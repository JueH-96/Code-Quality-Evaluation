s = input()
pos = {}
for i in range(26):
  pos[s[i]] = i + 1

ans = 0
curr = pos['A']
for char in 'BCDEFGHIJKLMNOPQRSTUVWXYZ':
  ans += abs(pos[char] - curr)
  curr = pos[char]

print(ans)