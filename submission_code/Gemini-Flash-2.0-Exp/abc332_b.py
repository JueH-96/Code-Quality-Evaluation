k, g, m = map(int, input().split())

glass = 0
mug = 0

for _ in range(k):
  if glass == g:
    glass = 0
  elif mug == 0:
    mug = m
  else:
    diff = g - glass
    if mug >= diff:
      glass += diff
      mug -= diff
    else:
      glass += mug
      mug = 0

print(glass, mug)