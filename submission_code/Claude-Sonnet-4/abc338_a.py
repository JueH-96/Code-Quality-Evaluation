# YOUR CODE HERE
S = input().strip()

# Check if first character is uppercase and all others are lowercase
if S[0].isupper() and S[1:].islower():
    print("Yes")
else:
    print("No")