# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
C = int(data[1])
T = list(map(int, data[2:]))

candies = 0
last_time = -1

for time in T:
    if time - last_time >= C:
        candies += 1
        last_time = time

print(candies)