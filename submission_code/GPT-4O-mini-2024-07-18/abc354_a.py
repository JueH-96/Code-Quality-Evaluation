H = int(input().strip())

day = 0
plant_height = 0

while plant_height <= H:
    day += 1
    plant_height = (1 << day) - 1  # This is equivalent to 2^day - 1

print(day)