# Read the input string from standard input
S = input()

# Check if all characters 'A', 'B', and 'C' are present in S
if all(char in S for char in "ABC"):
    print("Yes")
else:
    print("No")