# YOUR CODE HERE
numbers = []
while True:
    line = input()
    num = int(line)
    numbers.append(num)
    if num == 0:
        break

# Print the numbers in reverse order
for num in reversed(numbers):
    print(num)