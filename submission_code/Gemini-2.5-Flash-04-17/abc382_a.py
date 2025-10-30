# YOUR CODE HERE
import sys

# Read N and D from the first line
line1 = sys.stdin.readline().split()
n = int(line1[0])
d = int(line1[1])

# Read string S from the second line
s = sys.stdin.readline().strip()

# The state of the boxes is represented by string S.
# '.' denotes an empty box.
# '@' denotes a box with a cookie.

# Takahashi eats D cookies, one per day.
# Each cookie eaten makes the box it was in empty.
# Since he eats D cookies, D boxes that initially contained cookies will become empty.

# The boxes that were initially empty ('.') remain empty.
# The total number of empty boxes after D days is the sum of:
# 1. The number of boxes that were initially empty.
# 2. The number of boxes that become empty because their cookie was eaten.

# Number of initially empty boxes is the count of '.' in S.
initial_empty_count = s.count('.')

# Number of boxes that become empty by eating is equal to the number of cookies eaten, which is D.
# This is because each cookie is eaten from a distinct box that previously held a cookie.
boxes_becoming_empty_count = d

# Total number of empty boxes after D days is the sum of the above two counts.
total_empty_count = initial_empty_count + boxes_becoming_empty_count

# The problem guarantees that S contains at least D occurrences of '@', so there are always enough cookies for D days.
# The problem states the result is independent of the choice of cookies, confirming this simple calculation based on counts is valid.

# Print the final count of empty boxes
print(total_empty_count)