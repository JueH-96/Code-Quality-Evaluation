# YOUR CODE HERE
def solve():
  n = int(input())
  a = list(map(int, input().split()))
  
  count = 0
  for i in range(n):
    for j in range(i + 1, n):
      bi = 2**a[i]
      bj = 2**a[j]
      
      if (bi**bj) == (bj**bi):
        count += 1
  print(count)

t = int(input())
for _ in range(t):
  solve()