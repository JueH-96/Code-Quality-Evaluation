# YOUR CODE HERE
S = input().strip()

# Check if S can be rearranged to form "ABC"
# This is true if S contains exactly one 'A', one 'B', and one 'C'
if sorted(S) == sorted("ABC"):
    print("Yes")
else:
    print("No")