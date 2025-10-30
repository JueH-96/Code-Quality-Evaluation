S = input()
T = input()

if S == T:
  print(0)
else:
  min_len = min(len(S), len(T))
  for i in range(min_len):
    if S[i] != T[i]:
      print(i + 1)
      break
  else:
    print(max(len(S), len(T)))