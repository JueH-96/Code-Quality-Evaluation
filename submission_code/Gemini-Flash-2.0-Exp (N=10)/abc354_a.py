h = int(input())
plant_height = 0
day = 0
while True:
    plant_height += 2**day
    if plant_height > h:
        print(day + 1)
        break
    day += 1