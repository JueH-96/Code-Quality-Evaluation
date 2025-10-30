import sys
from collections import namedtuple
from itertools import permutations

Point = namedtuple('Point', 'x y')

def read_input():
    N = int(sys.stdin.readline())
    P = [Point(*map(int, sys.stdin.readline().split())) for _ in range(N)]
    Q = [Point(*map(int, sys.stdin.readline().split())) for _ in range(N)]
    return P, Q

def is_intersect(p1, p2, p3, p4):
    def ccw(p1, p2, p3):
        return (p2.x - p1.x)*(p3.y - p1.y) - (p2.y - p1.y)*(p3.x - p1.x)

    return ccw(p1, p2, p3)*ccw(p1, p2, p4) < 0 and ccw(p3, p4, p1)*ccw(p3, p4, p2) < 0

def solve(P, Q):
    for R in permutations(range(1, len(P) + 1)):
        if all(not is_intersect(P[i-1], Q[i-1], P[j-1], Q[j-1]) for i, j in permutations(R, 2)):
            return R
    return None

def main():
    P, Q = read_input()
    R = solve(P, Q)
    if R is None:
        print(-1)
    else:
        print(' '.join(map(str, R)))

if __name__ == '__main__':
    main()