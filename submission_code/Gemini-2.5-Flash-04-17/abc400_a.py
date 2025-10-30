# Read the input integer A
A = int(input())

# Check if 400 is perfectly divisible by A
if 400 % A == 0:
    # If divisible, calculate the corresponding integer B
    B = 400 // A
    # Print the value of B
    print(B)
else:
    # If not divisible, no such positive integer B exists, print -1
    print(-1)