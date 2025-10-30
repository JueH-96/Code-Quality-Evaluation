# YOUR CODE HERE
A, B, C = map(int, input().split())

# Possible sums of groups
total_sum = A + B + C

# Check if it's possible to divide into two or more groups with equal sums
# Possible group sums can be: A, B, C, A+B, B+C, C+A
# We need to check if any of these sums can be achieved by the other numbers

# Check if any of the combinations of two numbers equals the third number
if A + B == C or B + C == A or C + A == B:
    print("Yes")
# Check if all three numbers are the same
elif A == B == C:
    print("Yes")
else:
    print("No")