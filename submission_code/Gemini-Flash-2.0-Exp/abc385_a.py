A, B, C = map(int, input().split())

total_sum = A + B + C

if total_sum % 2 == 0:
    if (A == total_sum / 2) or (B == total_sum / 2) or (C == total_sum / 2) or (A + B == total_sum / 2) or (A + C == total_sum / 2) or (B + C == total_sum / 2):
        print("Yes")
    elif A == B and B == C:
        print("Yes")
    else:
        print("Yes")
else:
    print("No")