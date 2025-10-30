# YOUR CODE HERE
S = input().strip()

# Sort both strings and compare
if sorted(S) == sorted("ABC"):
    print("Yes")
else:
    print("No")