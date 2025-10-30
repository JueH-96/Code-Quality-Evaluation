# YOUR CODE HERE
S = input().strip()

# Check if S contains exactly the same characters as "ABC"
if sorted(S) == sorted("ABC"):
    print("Yes")
else:
    print("No")