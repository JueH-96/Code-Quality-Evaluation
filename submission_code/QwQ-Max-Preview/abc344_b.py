import sys

# Read all integers from input
numbers = []
for line in sys.stdin:
    num = int(line.strip())
    numbers.append(num)

# Reverse the list and print each element
for num in reversed(numbers):
    print(num)