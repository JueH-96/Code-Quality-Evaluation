import sys
sys.setrecursionlimit(1000000)

data = sys.stdin.read().split()
index = 0
N = int(data[index])
M = int(data[index+1])
Q = int(data[index+2])
index += 3

S = [0] * (M+1)
T = [0] * (M+1)
 for i in range(1, M+1):
  S[i] = int(data[index])
  T[i] = int(data[index+1])
  index += 2

L_min = [0] * (M+1)
R_max = [0] * (M+1)
 dir = [0] * (M+1)
 for i in range(1, M+1):
  if S[i] < T[i]:
   L_min[i] = S[i]
   R_max[i] = T[i]
   dir[i] = 1  # valley
  else:
   L_min[i] = T[i]
   R_max[i] = S[i]
   dir[i] = -1  # hill

# Read the queries
 queries = []
 for q in range(Q):
  L_k = int(data[index])
  R_k = int(data[index+1])
  index += 2
  queries.append((L_k, R_k))

# DSU class
 class DSU:
  def __init__(self, n):
   self.parent = [0] * (n+1)
   for i in range(1, n+1):
    self.parent[i] = i

  def find(self, x):
   if self.parent[x] != x:
    self.parent[x] = self.find(self.parent[x])
    return self.parent[x]
   return self.parent[x]

  def union(self, x, y):
   px = self.find(x)
   py = self.find(y)
   if px != py:
    self.parent[px] = py

 for L_k, R_k in queries:
  dsu = DSU(N)
  # Add unions for people from L_k to R_k
  for i in range(L_k, R_k+1):
   dsu.union(L_min[i], R_max[i])

  # Check for equality conflict
  conflict = False
  for i in range(L_k, R_k+1):
   L = L_min[i]
   R = R_max[i]
   for u in range(L+1, R):  # u from L+1 to R-1 inclusive
    if dsu.find(u) == dsu.find(L):
     conflict = True
     break
   if conflict:
    break

  if not conflict:
   # Check for inequality conflict with pairs
   for i in range(L_k, R_k+1):
    for j in range(i+1, R_k+1):
     if (dir[i] == dir[j] and not (L_min[i] <= L_min[j] and R_max[i] >= R_max[j] or L_min[j] <= L_min[i] and R_max[j] >= R_max[i]) and max(L_min[i], L_min[j]) < min(R_max[i], R_max[j])) or (L_min[i] == L_min[j] and dir[i] != dir[j]):
      conflict = True
      break
    if conflict:
     break

  if conflict:
   print("No")
  else:
   print("Yes")