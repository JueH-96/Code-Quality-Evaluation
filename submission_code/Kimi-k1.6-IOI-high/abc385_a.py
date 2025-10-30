A, B, C = map(int, input().split())
total = A + B + C
if (total % 2 == 0 and (A == total//2 or B == total//2 or C == total//2)) or (A == B == C):
    print("Yes")
else:
    print("No")