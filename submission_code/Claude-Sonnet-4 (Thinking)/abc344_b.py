# Read all integers until we encounter 0
numbers = []
while True:
    num = int(input())
    numbers.append(num)
    if num == 0:
        break

# Print in reverse order
for num in reversed(numbers):
    print(num)