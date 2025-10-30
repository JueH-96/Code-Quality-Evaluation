# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Find the largest number
max_num = max(A)

# Filter out the largest number and find the new maximum
second_max_num = max(filter(lambda x: x != max_num, A))

print(second_max_num)