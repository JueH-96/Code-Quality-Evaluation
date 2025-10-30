S = input()
flag = True
for i in range(1, 16, 2):
  if S[i] != '0':
    flag = False
    break

if flag:
  print("Yes")
else:
  print("No")