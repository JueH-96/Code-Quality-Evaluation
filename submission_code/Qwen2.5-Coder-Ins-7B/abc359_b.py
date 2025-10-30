# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

count = 0
for i in range(1, N + 1):
    indices = [j for j in range(2 * N) if A[j] == i]
    if len(indices) == 2 and abs(indices[0] - indices[1]) == 3:
        count += 1

print(count)