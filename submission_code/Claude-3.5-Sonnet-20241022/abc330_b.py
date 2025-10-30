N, L, R = map(int, input().split())
A = list(map(int, input().split()))

result = []
for a in A:
    # If a is less than L, closest value is L
    if a <= L:
        result.append(L)
    # If a is greater than R, closest value is R 
    elif a >= R:
        result.append(R)
    # If a is between L and R, closest value is a itself
    else:
        result.append(a)

print(*result)