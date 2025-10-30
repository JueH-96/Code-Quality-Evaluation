from collections import defaultdict

N = int(input())
ABs = [list(map(int, input().split())) for _ in range(N)]

group = defaultdict(list)
for i, (a, b) in enumerate(ABs):
  group[a].append((i, b))
  
count = [len(group[i]) for i in range(1,4)]
strength = [sum(b for i, b in group[i]) for i in range(1,4)]
total_strength = sum(strength)

if total_strength % 3:
  print(-1)
  exit(0)
else:
  target_strength = total_strength // 3

# i->j cost
costs = [[0] * 3 for _ in range(3)]

for si in range(3):
  for sj in range(3):
    if si == sj:
      continue
    
    temp = set()
    queues = [sorted(group[i+1], key=lambda x: -x[1])[count[(i+1)%3]//2:] for i in range(3)]
    for q in queues:
      for i, b in q:
        temp.add(i)

    count2 = [0] * 3
    strength2 = [0] * 3
    for i in range(3):
      for (j, b) in group[i+1]:
        if j in temp:
          strength2[i] += b
          count2[i] += 1
      
      if i == si:
        continue
      if i == sj:
        strength2[i] = target_strength - strength2[i]

    if min(strength2) * 2 >= max(strength2) and min(count2) * 2 >= max(count2):
      costs[si][sj] = sum(count2)

ans = float("inf")
# i->j and j->i
for si, sj in [(0, 1), (0, 2), (1, 2)]:
  ans = min(ans, costs[si][sj] + costs[sj][si])

# i->j->k and i->k
for i in range(3):
  for j in range(3):
    for k in range(3):
      if len(set([i, j, k])) < 3:
        continue
      ans = min(ans, costs[i][j] + costs[j][k])
      ans = min(ans, costs[i][j] + costs[i][k])
      
if ans == float("inf"):
  print(-1)
else:
  print(ans)