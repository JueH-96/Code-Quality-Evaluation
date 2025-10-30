S = input()
upper = 0
lower = 0
for i in S:
    if i.isupper():
        upper += 1
    else:
        lower += 1
if upper > lower:
    print(S.upper())
else:
    print(S.lower())