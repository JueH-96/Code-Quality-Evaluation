A, B, C = map(int, input().split())

# Check if all three are equal
if A == B == C:
    print("Yes")
# Check if any one is equal to the sum of the other two
elif A == B + C or B == A + C or C == A + B:
    print("Yes")
else:
    print("No")