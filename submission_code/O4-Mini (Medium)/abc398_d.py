import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    N = int(data[0])
    R = int(data[1])
    C = int(data[2])
    S = data[3]

    # Direction vectors
    mv = {'N': (-1,  0),
          'S': ( 1,  0),
          'W': ( 0, -1),
          'E': ( 0,  1)}

    # visited holds the set of prefix‐sum positions D[tau] for tau in {0} ∪ G up to current time
    visited = set()
    # initial prefix sum D[0] = (0,0)
    cx = 0
    cy = 0
    visited.add((0,0))

    res = []
    for ch in S:
        dx, dy = mv[ch]
        cx += dx
        cy += dy
        # We want to know if there exists tau< t with D[tau] = D[t] - (R,C)
        tx = cx - R
        ty = cy - C
        if (tx, ty) in visited:
            res.append('1')
        else:
            res.append('0')
        # Mark this prefix‐sum if first seen
        if (cx, cy) not in visited:
            visited.add((cx, cy))

    sys.stdout.write(''.join(res))

if __name__ == "__main__":
    main()