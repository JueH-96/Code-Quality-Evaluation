# YOUR CODE HERE
# Read input
L, R = map(int, input().split())

# Check the hand-raising scenario
if L == 1 and R == 0:
    # Only left hand raised = wants takoyaki
    print("Yes")
elif L == 0 and R == 1:
    # Only right hand raised = doesn't want takoyaki
    print("No")
else:
    # Both hands raised or no hands raised = invalid behavior
    print("Invalid")