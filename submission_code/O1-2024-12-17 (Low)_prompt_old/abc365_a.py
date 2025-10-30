def solve():
    Y = int(input().strip())
    if (Y % 400 == 0) or (Y % 4 == 0 and Y % 100 != 0):
        print(366)
    else:
        print(365)

solve()