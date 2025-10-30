# Read the input values for X and Y
X, Y = map(int, input().split())

# Check the conditions for using stairs
# Condition 1: Moving up two floors or less
condition_up_stairs = (Y > X and (Y - X) <= 2)

# Condition 2: Moving down three floors or less
condition_down_stairs = (X > Y and (X - Y) <= 3)

# If either condition for using stairs is true, print "Yes"
if condition_up_stairs or condition_down_stairs:
    print("Yes")
else:
    # Otherwise, Takahashi uses the elevator, print "No"
    print("No")