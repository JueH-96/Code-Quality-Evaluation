# Read the input values
A, B, C = map(int, input().split())

# Check if all three numbers are equal
if A == B == C:
    print("Yes")
else:
    total = A + B + C
    # Check if the sum is even
    if total % 2 != 0:
        print("No")
    else:
        target = total // 2
        # Check if any pair sums to target
        if (A + B == target) or (A + C == target) or (B + C == target):
            print("Yes")
        else:
            print("No")