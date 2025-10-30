# YOUR CODE HERE
A, B, C = map(int, input().split())

# Check if we can divide into two groups
# One group has one number, the other has two numbers
if A == B + C or B == A + C or C == A + B:
    print("Yes")
# Check if we can divide into three groups
# Each number is in its own group
elif A == B == C:
    print("Yes")
else:
    print("No")