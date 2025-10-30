H = int(input())

day = 0
height = 0
while height <= H:
    height += 2 ** day
    day += 1

print(day)