import sys
sys.setrecursionlimit(10000)

def main():
    input = sys.stdin.readline
    line = input().strip()
    if not line:
        return
    n = int(line)
    px = [0]*n
    py = [0]*n
    for i in range(n):
        a, b = map(int, input().split())
        px[i] = a
        py[i] = b
    qx = [0]*n
    qy = [0]*n
    for i in range(n):
        c, d = map(int, input().split())
        qx[i] = c
        qy[i] = d

    # Rmatch[i] = the index of Q matched to P_i
    Rmatch = [-1] * n

    def solve(red, blue):
        k = len(red)
        if k == 0:
            return True
        r0 = red[0]
        rx = px[r0]
        ry = py[r0]
        # Try matching r0 to some blue point bi
        for bi in blue:
            bx = qx[bi]
            by = qy[bi]
            dx_b = bx - rx
            dy_b = by - ry
            # count red points to left of the line r0->bi
            left_r = 0
            for rj in red:
                if rj == r0:
                    continue
                if dx_b * (py[rj] - ry) - dy_b * (px[rj] - rx) > 0:
                    left_r += 1
            # count blue points to left of the line r0->bi
            left_b = 0
            for bj in blue:
                if bj == bi:
                    continue
                if dx_b * (qy[bj] - ry) - dy_b * (qx[bj] - rx) > 0:
                    left_b += 1
            # if the left counts match, we can split and recurse
            if left_r == left_b:
                Rmatch[r0] = bi
                red_left = []
                red_right = []
                for rj in red:
                    if rj == r0:
                        continue
                    if dx_b * (py[rj] - ry) - dy_b * (px[rj] - rx) > 0:
                        red_left.append(rj)
                    else:
                        red_right.append(rj)
                blue_left = []
                blue_right = []
                for bj in blue:
                    if bj == bi:
                        continue
                    if dx_b * (qy[bj] - ry) - dy_b * (qx[bj] - rx) > 0:
                        blue_left.append(bj)
                    else:
                        blue_right.append(bj)
                if not solve(red_left, blue_left):
                    return False
                if not solve(red_right, blue_right):
                    return False
                return True
        return False

    ok = solve(list(range(n)), list(range(n)))
    if not ok:
        print(-1)
    else:
        # output 1-based indices
        print(" ".join(str(Rmatch[i] + 1) for i in range(n)))

if __name__ == "__main__":
    main()