A, B, C = map(int, input().split())

if B < C:
    if B <= A < C:
        print("No")
    else:
        print("Yes")
else:
    if (B <= A < 24) or (0 <= A < C):
        print("No")
    else:
        print("Yes")