import sys
import bisect

def main():
    H, W, Q = map(int, sys.stdin.readline().split())
    rows = [{} for _ in range(H+1)]
    cols = [{} for _ in range(W+1)]
    walls = set()
    
    for r in range(1, H+1):
        for c in range(1, W+1):
            walls.add((r, c))
            if r not in rows[r]:
                rows[r][c] = True
            if c not in cols[c]:
                cols[c][r] = True
    
    for _ in range(Q):
        R, C = map(int, sys.stdin.readline().split())
        if (R, C) in walls:
            walls.discard((R, C))
            if C in rows[R]:
                del rows[R][C]
            if R in cols[C]:
                del cols[C][R]
        else:
            # Find first wall in each direction
            # Up
            if C in cols[C]:
                keys = sorted(cols[C].keys())
                idx = bisect.bisect_left(keys, R)
                if idx > 0:
                    r_up = keys[idx-1]
                    if (r_up, C) in walls:
                        walls.discard((r_up, C))
                        del cols[C][r_up]
                        del rows[r_up][C]
            # Down
            if C in cols[C]:
                keys = sorted(cols[C].keys())
                idx = bisect.bisect_right(keys, R)
                if idx < len(keys):
                    r_down = keys[idx]
                    if (r_down, C) in walls:
                        walls.discard((r_down, C))
                        del cols[C][r_down]
                        del rows[r_down][C]
            # Left
            if R in rows[R]:
                keys = sorted(rows[R].keys())
                idx = bisect.bisect_left(keys, C)
                if idx > 0:
                    c_left = keys[idx-1]
                    if (R, c_left) in walls:
                        walls.discard((R, c_left))
                        del rows[R][c_left]
                        del cols[c_left][R]
            # Right
            if R in rows[R]:
                keys = sorted(rows[R].keys())
                idx = bisect.bisect_right(keys, C)
                if idx < len(keys):
                    c_right = keys[idx]
                    if (R, c_right) in walls:
                        walls.discard((R, c_right))
                        del rows[R][c_right]
                        del cols[c_right][R]
    
    print(len(walls))

if __name__ == "__main__":
    main()