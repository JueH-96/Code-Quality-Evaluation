numbers = []
while True:
    try:
        num = int(input())
        numbers.append(num)
        if num == 0:
            break
    except EOFError:
        break

for i in range(len(numbers) - 1, -1, -1):
    print(numbers[i])