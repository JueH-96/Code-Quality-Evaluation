y = int(input())
if y % 4 != 0:
    print(365)
else:
    if y % 100 != 0:
        print(366)
    else:
        if y % 400 != 0:
            print(365)
        else:
            print(366)