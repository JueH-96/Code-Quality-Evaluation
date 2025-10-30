# YOUR CODE HERE
S = input().strip()

# Sort the characters in both strings and compare
if sorted(S) == sorted("ABC"):
    print("Yes")
else:
    print("No")