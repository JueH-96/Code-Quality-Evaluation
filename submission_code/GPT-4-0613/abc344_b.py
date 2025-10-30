numbers = []
while True:
    number = int(input())
    numbers.append(number)
    if number == 0:
        break
for number in reversed(numbers):
    print(number)