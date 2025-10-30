import sys
data = sys.stdin.read().split()
N = int(data[0])
C = int(data[1])
T = [int(x) for x in data[2:2+N]]
candies = 1
last_time = T[0]
for t in T[1:]:
    if t - last_time >= C:
        candies += 1
        last_time = t
print(candies)