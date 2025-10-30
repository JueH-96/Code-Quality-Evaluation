H = int(input())

day = 1
plant_height = 1

while True:
    if plant_height > H:
        print(day)
        break
    day += 1
    plant_height = 2 * plant_height + 1