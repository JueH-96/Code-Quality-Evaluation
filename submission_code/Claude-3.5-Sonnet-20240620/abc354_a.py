# YOUR CODE HERE
H = int(input())

day = 0
plant_height = 0

while plant_height <= H:
    day += 1
    plant_height += 2**(day-1)

print(day)