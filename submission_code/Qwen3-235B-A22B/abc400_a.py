# Read the input from standard input
A = int(input())

# Check if A divides 400 evenly
if 400 % A == 0:
    B = 400 // A
    print(B)
else:
    print(-1)