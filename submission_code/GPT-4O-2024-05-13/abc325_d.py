# YOUR CODE HERE
import heapq
import sys

input = sys.stdin.read
data = input().split()

N = int(data[0])
products = []

for i in range(N):
    T = int(data[2 * i + 1])
    D = int(data[2 * i + 2])
    products.append((T, T + D))

# Sort products by their end time
products.sort(key=lambda x: x[1])

max_prints = 0
current_time = 0

for start, end in products:
    if current_time <= start:
        max_prints += 1
        current_time = end + 1

print(max_prints)