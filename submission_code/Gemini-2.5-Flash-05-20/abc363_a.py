# Read the input rating R
R = int(input())

# Initialize the variable to store the required increase
increase = 0

# Determine the current rating band and calculate the target rating for the next band
if 1 <= R <= 99:
    # If current rating is between 1 and 99 (inclusive), it displays 1 '^'.
    # To display 2 '^'s, the rating must be at least 100.
    # The minimum increase needed is the difference between 100 and the current rating.
    increase = 100 - R
elif 100 <= R <= 199:
    # If current rating is between 100 and 199 (inclusive), it displays 2 '^'s.
    # To display 3 '^'s, the rating must be at least 200.
    # The minimum increase needed is the difference between 200 and the current rating.
    increase = 200 - R
elif 200 <= R <= 299:
    # If current rating is between 200 and 299 (inclusive), it displays 3 '^'s.
    # To display 4 '^'s, the rating must be at least 300.
    # The minimum increase needed is the difference between 300 and the current rating.
    increase = 300 - R

# Print the calculated minimum required increase
print(increase)