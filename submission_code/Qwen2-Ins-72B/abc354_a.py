h = int(input())
day = 0
height = 0
while height <= h:
    day += 1
    height += 2 ** (day - 1)
print(day)