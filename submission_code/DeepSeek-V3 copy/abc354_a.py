# YOUR CODE HERE
H = int(input())
total_height = 0
day = 0
while True:
    if total_height > H:
        print(day)
        break
    total_height += 2 ** day
    day += 1