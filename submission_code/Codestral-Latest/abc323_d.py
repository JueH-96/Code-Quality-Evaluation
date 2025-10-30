import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
slimes = []

for i in range(1, 2*N, 2):
    S_i = int(data[i])
    C_i = int(data[i+1])
    slimes.append((S_i, C_i))

slimes.sort()

count = 0
for S_i, C_i in slimes:
    if C_i % 2 == 1:
        count += 1

print(count)