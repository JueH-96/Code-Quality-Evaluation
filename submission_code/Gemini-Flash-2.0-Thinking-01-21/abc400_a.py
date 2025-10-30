# Read the input integer A
A = int(input())

# Check if 400 is divisible by A
# If 400 is divisible by A, then B = 400 / A will be an integer.
# Since A is positive (1 <= A <= 400), 400 / A will also be positive.
if 400 % A == 0:
    # Calculate B using integer division
    B = 400 // A
    # Print the value of B
    print(B)
else:
    # If 400 is not divisible by A, there is no integer B
    # that satisfies A * B = 400.
    # Print -1 in this case.
    print(-1)