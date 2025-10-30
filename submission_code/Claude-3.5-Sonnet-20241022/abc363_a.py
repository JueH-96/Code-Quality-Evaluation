R = int(input())

if R <= 99:
    # Currently 1 ^, need to get to 100 for 2 ^
    print(100 - R)
elif R <= 199:
    # Currently 2 ^, need to get to 200 for 3 ^
    print(200 - R)
elif R <= 299:
    # Currently 3 ^, need to get to 300 for 4 ^
    print(300 - R)