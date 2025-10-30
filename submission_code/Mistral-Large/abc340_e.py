import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:2+N]))
B = list(map(int, data[2+N:]))

boxes = A[:]

for i in range(M):
    box_index = B[i]
    balls = boxes[box_index]
    boxes[box_index] = 0
    j = 1
    while balls > 0:
        next_box = (box_index + j) % N
        boxes[next_box] += 1
        balls -= 1
        j += 1

print(" ".join(map(str, boxes)))