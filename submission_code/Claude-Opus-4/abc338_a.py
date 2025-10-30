# YOUR CODE HERE
S = input().strip()

# Check if first character is uppercase
if not S[0].isupper():
    print("No")
else:
    # Check if all other characters are lowercase
    if all(c.islower() for c in S[1:]):
        print("Yes")
    else:
        print("No")