numbers = []
while True:
    line = input()
    num = int(line)
    numbers.append(num)
    if num == 0:
        break

numbers_reversed = numbers[::-1]
for num in numbers_reversed:
    print(num)