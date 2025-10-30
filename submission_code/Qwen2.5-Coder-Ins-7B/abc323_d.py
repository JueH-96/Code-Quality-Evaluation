# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
slimes = []

for i in range(N):
    S = int(data[2*i+1])
    C = int(data[2*i+2])
    slimes.append((S, C))

slimes.sort()

result = 0
for S, C in slimes:
    result += C // 2

print(result)