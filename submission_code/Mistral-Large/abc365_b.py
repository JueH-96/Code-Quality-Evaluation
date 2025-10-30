import sys

input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Find the index of the largest and second largest elements
largest_index = A.index(max(A))
largest_value = A.pop(largest_index)
second_largest_index = A.index(max(A))

# Adjust the index if the second largest is after the largest in the original list
if largest_index <= second_largest_index:
    second_largest_index += 1

print(second_largest_index + 1)