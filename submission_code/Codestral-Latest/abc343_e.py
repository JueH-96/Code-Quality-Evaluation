import sys
from itertools import product

def check_volumes(a1, b1, c1, a2, b2, c2, a3, b3, c3, V1, V2, V3):
    def intersect_volume(x1, x2, y1, y2, z1, z2):
        dx = max(0, min(x1 + 7, x2 + 7) - max(x1, x2))
        dy = max(0, min(y1 + 7, y2 + 7) - max(y1, y2))
        dz = max(0, min(z1 + 7, z2 + 7) - max(z1, z2))
        return dx * dy * dz

    def volume_in_one(c1, c2, c3):
        return (c1[1] - c1[0]) * (c1[3] - c1[2]) * (c1[5] - c1[4]) - intersect_volume(*c1, *c2) - intersect_volume(*c1, *c3) + intersect_volume(*c2, *c3)

    def volume_in_two(c1, c2, c3):
        return intersect_volume(*c1, *c2) + intersect_volume(*c1, *c3) + intersect_volume(*c2, *c3) - 2 * intersect_volume(*c1, *c2, *c3)

    def volume_in_three(c1, c2, c3):
        return intersect_volume(*c1, *c2, *c3)

    c1 = (a1, a1 + 7, b1, b1 + 7, c1, c1 + 7)
    c2 = (a2, a2 + 7, b2, b2 + 7, c2, c2 + 7)
    c3 = (a3, a3 + 7, b3, b3 + 7, c3, c3 + 7)

    return volume_in_one(c1, c2, c3) == V1 and volume_in_two(c1, c2, c3) == V2 and volume_in_three(c1, c2, c3) == V3

def find_solution(V1, V2, V3):
    for a1, b1, c1, a2, b2, c2, a3, b3, c3 in product(range(-100, 101), repeat=9):
        if check_volumes(a1, b1, c1, a2, b2, c2, a3, b3, c3, V1, V2, V3):
            return a1, b1, c1, a2, b2, c2, a3, b3, c3
    return None

def main():
    V1, V2, V3 = map(int, sys.stdin.read().split())
    solution = find_solution(V1, V2, V3)
    if solution:
        print("Yes")
        print(" ".join(map(str, solution)))
    else:
        print("No")

if __name__ == "__main__":
    main()