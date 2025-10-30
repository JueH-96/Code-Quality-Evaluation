# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
index = 1
A = []

for i in range(N):
    row = list(map(int, input[index:index + i + 1]))
    A.append(row)
    index += i + 1

current_element = 1
for i in range(1, N + 1):
    if current_element >= i:
        current_element = A[current_element - 1][i - 1]
    else:
        current_element = A[i - 1][current_element - 1]

print(current_element)