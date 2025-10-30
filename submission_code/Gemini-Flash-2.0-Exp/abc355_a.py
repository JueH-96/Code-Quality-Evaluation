A, B = map(int, input().split())

suspects = [1, 2, 3]
possible_culprits = suspects[:]

if A in possible_culprits:
  possible_culprits.remove(A)
if B in possible_culprits:
  possible_culprits.remove(B)

if len(possible_culprits) == 1:
  print(possible_culprits[0])
else:
  print("-1")