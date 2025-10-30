def popcount(x):
  count = 0
  while x > 0:
    count += x & 1
    x >>= 1
  return count

def solve():
  n, m = map(int, input().split())
  total_popcount = 0
  for k in range(n + 1):
    total_popcount = (total_popcount + popcount(k & m)) % 998244353
  print(total_popcount)

solve()