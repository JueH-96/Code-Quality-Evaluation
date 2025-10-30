import sys

input = sys.stdin.read
data = input().split()

N = int(data[0])
C = int(data[1])
T = list(map(int, data[2:]))

candies = 1
last_received = T[0]

for i in range(1, N):
    if T[i] - last_received >= C:
        candies += 1
        last_received = T[i]

print(candies)