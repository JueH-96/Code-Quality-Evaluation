# YOUR CODE HERE
# Read the current rating from standard input.
# The input is guaranteed to be an integer between 1 and 299.
R = int(input())

# The rating tiers are determined by hundreds.
# The number of '^'s increases when the rating crosses a hundred-multiple boundary (100, 200, 300, etc.).
# We need to find the minimum increase to reach the *next* boundary.

# Case 1: Current rating R is less than 100.
# The current tier is [1, 99].
# The next tier starts at 100.
# The minimum increase needed is 100 - R.
if R < 100:
    print(100 - R)

# Case 2: Current rating R is less than 200 (and we know R >= 100 from the first check).
# The current tier is [100, 199].
# The next tier starts at 200.
# The minimum increase needed is 200 - R.
elif R < 200:
    print(200 - R)

# Case 3: Current rating R is 200 or more (and up to 299 by the problem constraint).
# The current tier is [200, 299].
# The next tier starts at 300.
# The minimum increase needed is 300 - R.
else:
    print(300 - R)