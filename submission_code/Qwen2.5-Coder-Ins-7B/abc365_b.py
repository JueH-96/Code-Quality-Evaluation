# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

first_max = second_max = float('-inf')
first_max_index = second_max_index = -1

for i, a in enumerate(A):
    if a > first_max:
        second_max = first_max
        second_max_index = first_max_index
        first_max = a
        first_max_index = i
    elif a > second_max:
        second_max = a
        second_max_index = i

print(second_max_index + 1)