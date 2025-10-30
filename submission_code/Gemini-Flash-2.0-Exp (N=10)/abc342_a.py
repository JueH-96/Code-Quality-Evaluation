s = input()
if s[0] != s[1]:
  if s[0] == s[2]:
    print(2)
  else:
    print(1)
else:
  for i in range(1, len(s)):
    if s[i] != s[0]:
      print(i + 1)
      break