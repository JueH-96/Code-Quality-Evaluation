# YOUR CODE HERE
a1, a2, a3 = map(int, input().split())

# We need to check if any permutation (B_1, B_2, B_3) of (a1, a2, a3)
# satisfies the condition B_1 * B_2 = B_3.
# This is equivalent to checking if any two of the input numbers
# multiply together to equal the third input number.

# The possible pairs to multiply are (a1, a2), (a1, a3), and (a2, a3).
# The third numbers are a3, a2, and a1 respectively.

# Check if a1 * a2 equals a3
# Check if a1 * a3 equals a2
# Check if a2 * a3 equals a1

if a1 * a2 == a3 or \
   a1 * a3 == a2 or \
   a2 * a3 == a1:
    # If any of the conditions are true, it is possible
    print("Yes")
else:
    # If none of the conditions are true, it is not possible
    print("No")