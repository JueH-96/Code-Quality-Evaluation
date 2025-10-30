# Read the input string
S = input().strip()

# Check if the first character is uppercase and all other characters are lowercase
if S[0].isupper() and all(char.islower() for char in S[1:]):
    print("Yes")
else:
    print("No")