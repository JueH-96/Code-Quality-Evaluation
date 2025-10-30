import sys

# Read input from stdin
numbers = []
for line in sys.stdin:
    num = int(line.strip())
    numbers.append(num)
    if num == 0:
        break

# Print the numbers in reverse order
for num in reversed(numbers):
    print(num)