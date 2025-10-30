# Read N and D from the first line of input
N, D = map(int, input().split())

# Read the string S representing the state of the boxes
S = input()

# Calculate the number of boxes that are initially empty (contain '.')
# These boxes will remain empty throughout the D days.
initial_empty_boxes = S.count('.')

# Over D days, Takahashi eats D cookies.
# Each cookie eaten means one box that previously contained a cookie now becomes empty.
# These D boxes are distinct from the initially empty boxes, as they had cookies.
# Therefore, the total number of empty boxes after D days will be the sum of:
# 1. Boxes that were initially empty.
# 2. Boxes that contained cookies but became empty because their cookie was eaten.
# The number of such boxes is exactly D.

final_empty_boxes = initial_empty_boxes + D

# Print the result
print(final_empty_boxes)