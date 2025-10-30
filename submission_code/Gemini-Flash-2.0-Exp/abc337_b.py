S = input()
n = len(S)
a_end = -1
b_end = -1

for i in range(n):
  if S[i] != 'A':
    a_end = i
    break
else:
  a_end = n

if a_end == n:
  print("Yes")
  exit()

for i in range(a_end, n):
  if S[i] != 'B':
    b_end = i
    break
else:
  b_end = n

if b_end == n:
  print("Yes")
  exit()

for i in range(b_end, n):
  if S[i] != 'C':
    print("No")
    exit()

print("Yes")