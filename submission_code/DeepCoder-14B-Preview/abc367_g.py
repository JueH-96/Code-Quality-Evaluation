A, B, C = map(int, input().split())

if (B < C and (B <= A < C)) or (B > C and (A >= B or A < C)):
    print("No")
else:
    print("Yes")