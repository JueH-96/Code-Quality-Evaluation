# Read the input string from stdin
s = input()

# Check if the input string can be rearranged to match "ABC"
if sorted(s) == list("ABC"):
    print("Yes")
else:
    print("No")