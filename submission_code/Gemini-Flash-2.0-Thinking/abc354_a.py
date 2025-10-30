H = int(input())
day = 1
while True:
    plant_height = (2**day) - 1
    if plant_height > H:
        print(day)
        break
    day += 1