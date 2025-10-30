s = input()
t = input()

def dist(s1, s2):
  s1_ord = ord(s1) - ord('A')
  s2_ord = ord(s2) - ord('A')
  
  diff = abs(s1_ord - s2_ord)
  
  return min(diff, 5 - diff)

if dist(s[0], s[1]) == dist(t[0], t[1]):
  print("Yes")
else:
  print("No")