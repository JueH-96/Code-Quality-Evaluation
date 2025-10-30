N = int(input())
S = input()

if N % 2 == 0:
  print("No")
else:
  mid = (N + 1) // 2
  if S[mid - 1] != '/':
    print("No")
  else:
    flag = True
    for i in range(mid - 1):
      if i < 0 or i >= N:
        continue
      if S[i] != '1':
        flag = False
        break
    for i in range(mid, N):
      if i < 0 or i >= N:
        continue
      if S[i] != '2':
        flag = False
        break
    if flag:
      print("Yes")
    else:
      print("No")