R = int(input())
if R < 100:
    print(100 - R)
elif 100 <= R <= 199:
    print(200 - R)
else:
    print(300 - R)