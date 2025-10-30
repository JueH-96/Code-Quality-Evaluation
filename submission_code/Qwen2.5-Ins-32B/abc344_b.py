numbers = []
while True:
    try:
        num = int(input())
        numbers.append(num)
        if num == 0:
            break
    except EOFError:
        break

for num in reversed(numbers):
    print(num)