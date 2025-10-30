# YOUR CODE HERE
numbers = []
while True:
    n = int(input())
    numbers.append(n)
    if n == 0:
        break

# Print in reverse order
for i in range(len(numbers) - 1, -1, -1):
    print(numbers[i])