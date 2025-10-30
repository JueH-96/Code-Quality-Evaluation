# YOUR CODE HERE
numbers = []
while True:
    num = int(input())
    numbers.append(num)
    if num == 0:
        break

# Print in reverse order
for i in range(len(numbers) - 1, -1, -1):
    print(numbers[i])