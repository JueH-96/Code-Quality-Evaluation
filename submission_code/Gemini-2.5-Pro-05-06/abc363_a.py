# YOUR CODE HERE
R = int(input())

increase = 0
if R < 100: # This covers the range 1 <= R <= 99
    # Current: 1 ^. Target rating for 2 ^ is 100.
    increase = 100 - R
elif R < 200: # This covers the range 100 <= R <= 199
    # Current: 2 ^. Target rating for 3 ^ is 200.
    increase = 200 - R
else: # Given R <= 299, this covers the range 200 <= R <= 299
    # Current: 3 ^. Target rating for 4 ^ is 300.
    increase = 300 - R

print(increase)