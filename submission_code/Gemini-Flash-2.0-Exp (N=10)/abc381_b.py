s = input()
n = len(s)

if n % 2 != 0:
  print("No")
else:
  is_1122 = True
  for i in range(n // 2):
    if s[2 * i] != s[2 * i + 1]:
      is_1122 = False
      break
  
  if is_1122:
    counts = {}
    for char in s:
      counts[char] = counts.get(char, 0) + 1
    
    for count in counts.values():
      if count != 2:
        is_1122 = False
        break
  
  if is_1122:
    print("Yes")
  else:
    print("No")