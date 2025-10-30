# Read the input values
V1, V2, V3 = map(int, input().split())

# Check the total volume condition
total = V1 + 2 * V2 + 3 * V3

if total != 1029:
    print("No")
else:
    # Check if V3 is less than or equal to V2 and V2 is less than or equal to V1
    if V3 <= V2 and V2 <= V1:
        print("Yes")
    else:
        print("No")