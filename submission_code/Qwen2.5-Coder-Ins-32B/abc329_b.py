# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
A = list(map(int, input[1:]))

# Find the largest number
max_num = max(A)

# Filter out the largest number and find the largest among the rest
second_max_num = max(num for num in A if num != max_num)

print(second_max_num)