R = int(input())

if R <= 99:
    target = 100
elif R <= 199:
    target = 200
else:  # R <= 299
    target = 300

increase = target - R
print(increase)