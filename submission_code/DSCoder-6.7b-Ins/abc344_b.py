# YOUR CODE HERE
import sys

numbers = []
for line in sys.stdin:
    numbers.append(int(line.strip()))

for number in reversed(numbers):
    print(number)