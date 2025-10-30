# YOUR CODE HERE
R = int(input())

if R < 100:
    print(100 - R)
elif R < 200:
    print(200 - R)
else:  # R < 300 (since R <= 299)
    print(300 - R)