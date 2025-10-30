A, B, C = map(int, input().split())
total = A + B + C

if total % 2 == 0:
    target = total // 2
    if (A == target or B == target or C == target or
        (A + B == target) or (A + C == target) or (B + C == target)):
        print("Yes")
        exit()

if A == B == C:
    print("Yes")
else:
    print("No")