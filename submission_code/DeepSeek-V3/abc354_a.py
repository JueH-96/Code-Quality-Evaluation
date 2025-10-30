# YOUR CODE HERE
H = int(input())
day = 0
height = 0
while True:
    if height > H:
        print(day)
        break
    height += 2 ** day
    day += 1