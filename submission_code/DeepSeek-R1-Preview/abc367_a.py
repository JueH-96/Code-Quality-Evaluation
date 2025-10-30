A, B, C = map(int, input().split())

if B < C:
    if A < B or A >= C:
        print("Yes")
    else:
        print("No")
else:
    if C <= A < B:
        print("Yes")
    else:
        print("No")