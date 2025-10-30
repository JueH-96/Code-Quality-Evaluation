# Read the input string
S = input().strip()

# Check if the first character is uppercase and the rest are lowercase
if S[0].isupper() and S[1:].islower():
    print("Yes")
else:
    print("No")