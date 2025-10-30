def grundy(n):
  if n == 1:
    return 0
  
  divisors = []
  for i in range(1, int(n**0.5) + 1):
    if n % i == 0:
      if i != n:
        divisors.append(i)
      if n // i != n and i * i != n:
        divisors.append(n // i)
  
  divisors.sort()
  
  mex_set = set()
  for d in divisors:
    mex_set.add(grundy(d))
  
  mex = 0
  while mex in mex_set:
    mex += 1
  return mex

def solve():
  n = int(input())
  a = list(map(int, input().split()))
  
  xor_sum = 0
  for x in a:
    xor_sum ^= grundy(x)
  
  if xor_sum == 0:
    print("Bruno")
  else:
    print("Anna")

solve()