# YOUR CODE HERE
S = input().strip()

# Check if the string S can be rearranged to match "ABC"
if sorted(S) == ['A', 'B', 'C']:
    print("Yes")
else:
    print("No")