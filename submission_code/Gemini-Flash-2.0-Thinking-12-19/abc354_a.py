H = int(input())
day = 1
power_of_2 = 2
while True:
    current_height = power_of_2 - 1
    if current_height > H:
        print(day)
        break
    power_of_2 *= 2
    day += 1