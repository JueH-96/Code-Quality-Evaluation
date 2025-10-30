# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
days = list(map(int, data[1:]))

count = 0
for i in range(1, N + 1):
    for j in range(1, days[i - 1] + 1):
        if str(i) == str(j):
            count += 1

print(count)