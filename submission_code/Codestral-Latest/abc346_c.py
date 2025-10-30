# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
A = list(map(int, data[2:]))

# Create a set of numbers in A for quick lookup
A_set = set(A)

# Calculate the sum of numbers from 1 to K that are not in A
missing_sum = sum(i for i in range(1, K + 1) if i not in A_set)

print(missing_sum)