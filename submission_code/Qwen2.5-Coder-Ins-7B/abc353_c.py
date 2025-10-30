# YOUR CODE HERE
import sys

input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

total_sum = 0

for i in range(N-1):
    for j in range(i+1, N):
        total_sum += (A[i] + A[j]) % 100000000

print(total_sum)