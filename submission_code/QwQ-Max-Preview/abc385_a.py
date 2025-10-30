# Read the input values
A, B, C = map(int, input().split())

sum_total = A + B + C

# Check if all three are equal
if A == B == C:
    print("Yes")
else:
    # Check if the sum is even
    if sum_total % 2 == 0:
        target = sum_total // 2
        # Check if any single number is the target or any two sum to target
        if (A == target or B == target or C == target) or (A + B == target or A + C == target or B + C == target):
            print("Yes")
        else:
            print("No")
    else:
        print("No")