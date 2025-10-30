s = input()
n = len(s)
if n == 0:
  print("Yes")
else:
  i = 0
  while i < n and s[i] == 'A':
    i += 1
  j = i
  while j < n and s[j] == 'B':
    j += 1
  k = j
  while k < n and s[k] == 'C':
    k += 1
  if k == n and j > i and i >= 0:
    print("Yes")
  else:
    print("No")