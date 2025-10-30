import sys
from collections import defaultdict

def read_input():
    N = int(sys.stdin.readline().strip())
    cuboids = []
    for _ in range(N):
        x1, y1, z1, x2, y2, z2 = map(int, sys.stdin.readline().strip().split())
        cuboids.append(((x1, y1, z1), (x2, y2, z2)))
    return cuboids

def solve(cuboids):
    faces = defaultdict(int)
    for ((x1, y1, z1), (x2, y2, z2)) in cuboids:
        for (x, y, z) in [(x1, y1, z1), (x1, y1, z2), (x1, y2, z1), (x1, y2, z2),
                          (x2, y1, z1), (x2, y1, z2), (x2, y2, z1), (x2, y2, z2)]:
            faces[(x, y, z)] += 1
    return [faces[face] - 1 for ((x1, y1, z1), (x2, y2, z2)) in cuboids for face in 
            [(x1, y1, z1), (x1, y1, z2), (x1, y2, z1), (x1, y2, z2),
             (x2, y1, z1), (x2, y1, z2), (x2, y2, z1), (x2, y2, z2)]]

def main():
    cuboids = read_input()
    result = solve(cuboids)
    print('
'.join(map(str, result)))

if __name__ == "__main__":
    main()