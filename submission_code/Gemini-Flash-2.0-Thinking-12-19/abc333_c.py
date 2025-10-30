def repunit(k):
  if k <= 0:
    return 0
  return int('1' * k)

sums_list = []
for i in range(1, 13):
  for j in range(i, 13):
    for k in range(j, 13):
      sum_val = repunit(i) + repunit(j) + repunit(k)
      sums_list.append(sum_val)

sums_list.sort()
n = int(input())
print(sums_list[n-1])