# YOUR CODE HERE
import sys
input = sys.stdin.read

data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))

# Create a list of tuples (value, index)
indexed_A = [(A[i], i + 1) for i in range(N)]

# Sort the list by value in descending order
sorted_A = sorted(indexed_A, key=lambda x: x[0], reverse=True)

# The second element in the sorted list is the second largest
second_largest_index = sorted_A[1][1]

print(second_largest_index)