# YOUR CODE HERE
W, B = map(int, input().split())

# The string S is repeated infinitely
# The string S is wbwbwwbwbwbw
# The length of S is 12

# If W + B is greater than or equal to 12, then there is a substring of S that consists of W occurrences of w and B occurrences of b
if W + B >= 12:
    print("Yes")
else:
    print("No")