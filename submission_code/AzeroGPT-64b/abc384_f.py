import collections
N = int(input())
A = [int(a) for a in input().split()]
X = 30
P = [1 << x for x in range(31)]
pow2 = dict()
for p in P:
  pow2[p] = 0
for a in A:
  x = 1
  while a >= x * 2:
    x *= 2
  pow2[x] += 1
cnt = [collections.Counter(), collections.Counter()]
for p, c in pow2.items():
  for x in P:
    if (p + x) in P and (p <= x*(1<<X) - x):
      cnt[(p + x) in P][x] += c
ans = 0
for bits, n in zip(cnt, [1, 2]):
  for p in P:
    x = p
    while x > 1:
      x >>= 1
      ans += bits[x] * n
print(ans)