# YOUR CODE HERE
H = int(input())

day = 0
plant_height = 0

while plant_height <= H:
    plant_height += 2 ** day
    day += 1

print(day)