import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
cuboids = []
index = 1

for i in range(N):
    x1, y1, z1, x2, y2, z2 = map(int, data[index:index+6])
    cuboids.append(((x1, y1, z1, x2, y2, z2)))
    index += 6

def shares_face(c1, c2):
    x1_min, y1_min, z1_min, x1_max, y1_max, z1_max = c1
    x2_min, y2_min, z2_min, x2_max, y2_max, z2_max = c2

    if (x1_max == x2_min and y1_min <= y2_max and y1_max >= y2_min and z1_min <= z2_max and z1_max >= z2_min) or \
       (y1_max == y2_min and x1_min <= x2_max and x1_max >= x2_min and z1_min <= z2_max and z1_max >= z2_min) or \
       (z1_max == z2_min and x1_min <= x2_max and x1_max >= x2_min and y1_min <= y2_max and y1_max >= y2_min):
        return True
    return False

result = [0] * N

for i in range(N):
    for j in range(i + 1, N):
        if shares_face(cuboids[i], cuboids[j]):
            result[i] += 1
            result[j] += 1

for count in result:
    print(count)