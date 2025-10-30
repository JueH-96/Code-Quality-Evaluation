A, B, C = map(int, input().split())
if B < C:
    if A >= B and A < C:
        print("No")
    else:
        print("Yes")
else:
    if A >= B or A < C:
        print("Yes")
    else:
        print("No")