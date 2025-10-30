# YOUR CODE HERE
numbers = []
while True:
    try:
        num = int(input())
        if num == 0:
            numbers.append(num)
            break
        numbers.append(num)
    except EOFError:
        break

for num in reversed(numbers):
    print(num)