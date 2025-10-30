N, Q = map(int, input().split())
T = list(map(int, input().split()))

teeth = [1] * N
count = N

for t in T:
  t -= 1
  if teeth[t] == 1:
    teeth[t] = 0
    count -= 1
  else:
    teeth[t] = 1
    count += 1

print(count)