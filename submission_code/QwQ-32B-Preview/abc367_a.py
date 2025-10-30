A, B, C = map(int, input().split())

if B < C:
    if not (B <= A < C):
        print("Yes")
    else:
        print("No")
else:
    if not (B <= A < 24 or 0 <= A < C):
        print("Yes")
    else:
        print("No")