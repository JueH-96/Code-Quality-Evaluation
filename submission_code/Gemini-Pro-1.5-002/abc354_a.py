# YOUR CODE HERE
h = int(input())
plant_height = 0
days = 0
while plant_height <= h:
    days += 1
    plant_height = (1 << days) - 1

print(days)