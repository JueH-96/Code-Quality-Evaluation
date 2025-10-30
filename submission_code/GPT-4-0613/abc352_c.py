import sys
from operator import itemgetter

N = int(sys.stdin.readline())
giants = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]

giants.sort(key=itemgetter(0, 1), reverse=True)

shoulder_height = giants[0][0]
max_height = giants[0][0] + giants[0][1]

for i in range(1, N):
    shoulder_height += giants[i][0]
    max_height = max(max_height, shoulder_height + giants[i][1])

print(max_height)