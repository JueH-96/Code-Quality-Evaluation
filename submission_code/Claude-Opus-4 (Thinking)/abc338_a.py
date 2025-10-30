# YOUR CODE HERE
S = input()

# Check if first character is uppercase
if S[0].isupper():
    # Check if all other characters (if any) are lowercase
    if all(c.islower() for c in S[1:]):
        print("Yes")
    else:
        print("No")
else:
    print("No")