N = int(input())
R = input()
C = input()
 
def duplicate(s):
  return len(s) != len(set(s))
 
if duplicate(R) or duplicate(C):
  print('No')
  exit()
 
lst = [list('.' * N) for _ in range(N)]
 
for i in range(N):
  lst[i][0] = R[i]
  lst[0][i] = C[i]
 
flg = True
for i in range(1, N):
  for j in range(1, N):
    if lst[i][j] == '.':
      if lst[i][0] == lst[0][j]:
        lst[i][j] = lst[i][0]
      else:
        lst[i][j] = 'C'
        if lst[i][j] == lst[i][0] or lst[i][j] == lst[0][j]:
          lst[i][j] = 'B'
          if lst[i][j] == lst[i][0] or lst[i][j] == lst[0][j]:
            print('No')
            exit(0)
 
for i in range(1, N):
  for j in range(1, N):
    if lst[i][j] == 'A':
      key = lst[i][0]
      if key == 'B':
        lst[i+1][(j+2) % 3] = 'B'
        lst[i+1][(j+1) % 3] = 'C'
      elif key == 'C':
        lst[i+1][(j+2) % 3] = 'C'
        lst[i+1][(j+1) % 3] = 'B'
 
      key = lst[0][j]
      if key == 'B':
        lst[(i+2) % 3][j] = 'B'
        lst[(i+1) % 3][j] = 'C'
      elif key == 'C':
        lst[(i+2) % 3][j] = 'C'
        lst[(i+1) % 3][j] = 'B'
 
    elif lst[i][j] == 'B':
      key = lst[0][j]
      if key == 'C':
        lst[i+1][(j+2) % 3] = 'C'
        lst[i+1][(j+1) % 3] = 'A'
      elif key == 'A':
        lst[i+1][(j+2) % 3] = 'A'
        lst[i+1][(j+1) % 3] = 'C'
 
      key = lst[i][0]
      if key == 'C':
        lst[(i+2) % 3][j] = 'C'
        lst[(i+1) % 3][j] = 'A'
      elif key == 'A':
        lst[(i+2) % 3][j] = 'A'
        lst[(i+1) % 3][j] = 'C'
 
    elif lst[i][j] == 'C':
      key = lst[i][0]
      if key == 'B':
        lst[i+1][(j+2) % 3] = 'B'
        lst[i+1][(j+1) % 3] = 'A'
      elif key == 'A':
        lst[i+1][(j+2) % 3] = 'A'
        lst[i+1][(j+1) % 3] = 'B'
 
      key = lst[0][j]
      if key == 'B':
        lst[(i+2) % 3][j] = 'B'
        lst[(i+1) % 3][j] = 'A'
      elif key == 'A':
        lst[(i+2) % 3][j] = 'A'
        lst[(i+1) % 3][j] = 'B'
print('Yes')
for a in lst:
  print(''.join(a))