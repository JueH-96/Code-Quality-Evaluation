# Read the input values for L and R
L, R = map(int, input().split())

# Calculate the sum of L and R
sum_hands = L + R

# Determine the output based on the sum and individual values
if sum_hands == 1:
    if L == 1:
        print("Yes")
    else:
        print("No")
else:
    print("Invalid")