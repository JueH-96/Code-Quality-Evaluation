a, b = map(int, input().split())

possible_culprits = {1, 2, 3}

if a in possible_culprits:
    possible_culprits.discard(a)
if b in possible_culprits:
    possible_culprits.discard(b)

if len(possible_culprits) == 1:
    print(possible_culprits.pop())
else:
    print(-1)