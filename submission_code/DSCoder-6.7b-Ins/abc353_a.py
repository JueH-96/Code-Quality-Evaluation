# YOUR CODE HERE

import sys

N = int(sys.stdin.readline().strip())
H = list(map(int, sys.stdin.readline().strip().split()))

first_building_height = H[0]
taller_building_position = -1

for i in range(1, N):
    if H[i] > first_building_height:
        taller_building_position = i
        break

print(taller_building_position)