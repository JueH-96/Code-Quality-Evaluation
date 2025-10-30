R = int(input())

if 1 <= R <= 99:
    # Currently 1 ^, need to reach 100 for 2 ^s
    print(100 - R)
elif 100 <= R <= 199:
    # Currently 2 ^s, need to reach 200 for 3 ^s
    print(200 - R)
elif 200 <= R <= 299:
    # Currently 3 ^s, need to reach 300 for 4 ^s
    print(300 - R)