# Read the input values
A, B = map(int, input().split())

# Check if B is exactly the next number after A
if B == A + 1:
    # Check if A is not a multiple of 3 (so they are in the same row)
    if A % 3 != 0:
        print("Yes")
    else:
        print("No")
else:
    print("No")