H = int(input())
height = 0
day = 0
while height <= H:
    height += 2**day
    day += 1
print(day)