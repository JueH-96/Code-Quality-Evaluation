# YOUR CODE HERE
H = int(input())
height = 0
day = 0
while height <= H:
    day += 1
    height += 2**(day-1)
print(day)