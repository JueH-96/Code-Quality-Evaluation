# YOUR CODE HERE
R = int(input())

if R <= 99:
    next_threshold = 100
elif R <= 199:
    next_threshold = 200
else:  # R <= 299
    next_threshold = 300

increase = next_threshold - R
print(increase)