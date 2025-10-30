# Read the input rating R
R = int(input())

# Determine the minimum increase required to reach the next rating threshold
if R >= 1 and R <= 99:
    # Current rating is in the 1-99 range (1 ^).
    # The next threshold is 100.
    increase = 100 - R
elif R >= 100 and R <= 199:
    # Current rating is in the 100-199 range (2 ^).
    # The next threshold is 200.
    increase = 200 - R
elif R >= 200 and R <= 299:
    # Current rating is in the 200-299 range (3 ^).
    # The next threshold is 300.
    increase = 300 - R
# The problem guarantees that R is between 1 and 299 inclusive,
# so these three cases cover all possible inputs.

# Print the calculated minimum increase
print(increase)