# YOUR CODE HERE
A, B, C = map(int, input().split())

# Check if any number equals the sum of the other two, or if all three are equal
if A == B + C or B == A + C or C == A + B or (A == B == C):
    print("Yes")
else:
    print("No")