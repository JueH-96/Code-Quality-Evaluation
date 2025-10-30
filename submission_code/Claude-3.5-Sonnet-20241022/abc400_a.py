# Read input
A = int(input())

# Check if 400 is divisible by A
if 400 % A == 0:
    # If divisible, B is 400/A
    B = 400 // A
    print(B)
else:
    # If not divisible, no rectangular arrangement is possible
    print(-1)