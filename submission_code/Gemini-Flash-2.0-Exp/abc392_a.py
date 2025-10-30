from itertools import permutations

a, b, c = map(int, input().split())

arr = [a, b, c]

perms = permutations(arr)

for p in perms:
  if p[0] * p[1] == p[2]:
    print("Yes")
    exit()

print("No")