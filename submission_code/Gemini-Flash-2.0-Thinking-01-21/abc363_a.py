R = int(input())

if 1 <= R <= 99:
    increase = 100 - R
elif 100 <= R <= 199:
    increase = 200 - R
elif 200 <= R <= 299:
    increase = 300 - R
else:
    increase = 0 # This case should not happen given the problem constraints

print(increase)