# YOUR CODE HERE
W, B = map(int, input().split())

S = "wbwbwwbwbwbw"
long_S = ""
for i in range(100):
  long_S += S

found = False
for i in range(len(long_S)):
  for j in range(i + 1, len(long_S) + 1):
    sub = long_S[i:j]
    w_count = sub.count('w')
    b_count = sub.count('b')
    if w_count == W and b_count == B:
      found = True
      break
  if found:
    break

if found:
  print("Yes")
else:
  print("No")