A, B, C = map(int, input().split())

if A == B + C or B == A + C or C == A + B or (A == B and B == C):
    print("Yes")
else:
    print("No")