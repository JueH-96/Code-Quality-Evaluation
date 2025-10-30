H = int(input())
plant_height = 0
day = 0
while plant_height <= H:
    day += 1
    plant_height += 2**(day - 1)
print(day)