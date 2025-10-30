A, B = map(int, input().split())
attacks = A // B
if A % B != 0:
    attacks += 1
print(attacks)