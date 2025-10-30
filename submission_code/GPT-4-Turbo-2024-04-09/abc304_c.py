import sys
import math

input = sys.stdin.read
data = input().split()

N = int(data[0])
D = int(data[1])
coordinates = [(int(data[i*2+2]), int(data[i*2+3])) for i in range(N)]

def is_infected(start, D, coordinates):
    infected = [False] * len(coordinates)
    infected[start] = True
    queue = [start]
    
    while queue:
        current = queue.pop(0)
        for i in range(len(coordinates)):
            if not infected[i]:
                if math.sqrt((coordinates[current][0] - coordinates[i][0]) ** 2 + (coordinates[current][1] - coordinates[i][1]) ** 2) <= D:
                    infected[i] = True
                    queue.append(i)
    
    return infected

infected_result = is_infected(0, D, coordinates)

for result in infected_result:
    if result:
        print("Yes")
    else:
        print("No")