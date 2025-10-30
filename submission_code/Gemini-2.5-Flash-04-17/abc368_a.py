# YOUR CODE HERE
import sys

# Read N (number of cards) and K (number of cards to move from bottom to top)
# N and K are on the first line, separated by space.
line1 = sys.stdin.readline().split()
n = int(line1[0])
k = int(line1[1])

# Read the list of N integers representing the cards from top to bottom.
# The integers are on the second line, separated by spaces.
line2 = sys.stdin.readline().split()
a = list(map(int, line2))

# The list `a` represents the stack: a[0] is the top, a[n-1] is the bottom.
# We need to take the bottom K cards and move them to the top,
# preserving their relative order.
# The bottom K cards are the last K elements of the list.
# In Python slicing, `a[-k:]` gives the slice containing the last K elements.
# The remaining cards are the first N-K elements of the original list.
# In Python slicing, `a[:-k]` gives the slice containing all elements except the last K.
# The new order places the bottom K cards on top, followed by the remaining cards.
# This new order from top to bottom is achieved by concatenating the two list slices.
rearranged_a = a[-k:] + a[:-k]

# Print the elements of the rearranged list.
# The requirement is to print them separated by spaces.
# The `*` operator unpacks the elements of the list `rearranged_a` into individual arguments
# for the `print` function.
# The `print` function, by default, separates its arguments with a space.
# This achieves the desired output format.
print(*rearranged_a)