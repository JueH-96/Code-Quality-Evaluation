a, b = map(int, input().split())

suspects = [1, 2, 3]
possible_culprits = []

for suspect in suspects:
    if suspect != a and suspect != b:
        possible_culprits.append(suspect)
    elif suspect != a and b == a:
        possible_culprits.append(suspect)
    elif suspect != b and a == b:
        possible_culprits.append(suspect)
    elif a == b and suspect != a:
        possible_culprits.append(suspect)

if len(possible_culprits) == 1:
    print(possible_culprits[0])
else:
    print(-1)