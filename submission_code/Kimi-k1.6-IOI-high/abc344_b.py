# Read all numbers until a zero is encountered
numbers = []
while True:
    num = int(input().strip())
    numbers.append(num)
    if num == 0:
        break

# Print the numbers in reverse order
for num in reversed(numbers):
    print(num)