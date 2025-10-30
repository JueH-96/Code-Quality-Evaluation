import sys

# Read input
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Sort the sellers and buyers
A.sort()
B.sort()

# Find the minimum price that satisfies the condition
price = 0
i = 0
j = 0
while i < N and j < M:
    if A[i] <= B[j]:
        price = A[i]
        break
    i += 1
    j += 1

# If no price satisfies the condition, return the maximum possible price
if price == 0:
    price = max(A[-1], B[-1]) + 1

print(price)