import sys
from itertools import combinations

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

A = list(map(int, data[2:2+N]))
B = list(map(int, data[2+N:2+N+M]))

# Sort the boxes by the number of candies in ascending order
A.sort()

# Sort the people by the minimum number of candies they need in ascending order
B.sort()

# Check if it's possible to satisfy the condition
if any(A[i] < B[i] for i in range(M)):
    print(-1)
else:
    # Calculate the minimum total amount of money Takahashi needs to pay
    min_cost = sum(A[i] for i in range(M))
    print(min_cost)