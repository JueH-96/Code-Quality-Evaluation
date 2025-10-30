R = int(input())
if R <= 99:
    current = 1
elif R <= 199:
    current = 2
else:
    current = 3
next_threshold = current * 100
print(next_threshold - R)