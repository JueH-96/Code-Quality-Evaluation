# YOUR CODE HERE

# Read the inputs
A, B, C = map(int, input().split())

# Check if Takahashi can shout his love for takoyaki every day
if A < B < C or C < B < A:
    print("Yes")
else:
    print("No")