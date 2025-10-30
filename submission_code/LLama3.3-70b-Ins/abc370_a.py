# Read input from stdin
L, R = map(int, input().split())

# Check the conditions and print the result
if L == 1 and R == 0:
    print("Yes")
elif L == 0 and R == 1:
    print("No")
else:
    print("Invalid")