# YOUR CODE HERE
H = int(input())

day = 1
while (2**day - 1) <= H:
    day += 1

print(day)