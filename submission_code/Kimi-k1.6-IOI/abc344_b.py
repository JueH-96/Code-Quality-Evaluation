numbers = []
while True:
    num = int(input().strip())
    numbers.append(num)
    if num == 0:
        break
for num in reversed(numbers):
    print(num)