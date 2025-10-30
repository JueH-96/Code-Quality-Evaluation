S = input()
n = len(S)

if n % 2 != 0:
  print("No")
else:
  flag = True
  for i in range(n // 2):
    if S[2 * i] != S[2 * i + 1]:
      flag = False
      break
  
  counts = {}
  for char in S:
    if char in counts:
      counts[char] += 1
    else:
      counts[char] = 1
      
  for char in counts:
    if counts[char] != 2:
      flag = False
      break
      
  if flag:
    print("Yes")
  else:
    print("No")