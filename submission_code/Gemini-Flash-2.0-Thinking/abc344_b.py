numbers = []
while True:
    try:
        line = input()
        num = int(line)
        numbers.append(num)
        if num == 0:
            break
    except EOFError:
        break

for num in reversed(numbers):
    print(num)