# YOUR CODE HERE
import sys

input = sys.stdin.read().strip().split('
')
numbers = [int(num) for num in input]
numbers.reverse()

for num in numbers:
    print(num)