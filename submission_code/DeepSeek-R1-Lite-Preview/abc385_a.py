A, B, C = map(int, input().split())
S = A + B + C

# Check for two groups with equal sum
if S % 2 == 0:
    half = S / 2
    if (A + B == half) or (A + C == half) or (B + C == half):
        print("Yes")
        exit()

# Check for three groups with equal sum
if A == B == C:
    print("Yes")
    exit()

print("No")