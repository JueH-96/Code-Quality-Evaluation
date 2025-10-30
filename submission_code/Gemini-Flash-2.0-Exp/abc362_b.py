def solve():
    xa, ya = map(int, input().split())
    xb, yb = map(int, input().split())
    xc, yc = map(int, input().split())

    def dist_sq(x1, y1, x2, y2):
        return (x1 - x2)**2 + (y1 - y2)**2

    ab_sq = dist_sq(xa, ya, xb, yb)
    bc_sq = dist_sq(xb, yb, xc, yc)
    ca_sq = dist_sq(xc, yc, xa, ya)

    sides = sorted([ab_sq, bc_sq, ca_sq])

    if sides[0] + sides[1] == sides[2]:
        print("Yes")
    else:
        print("No")

solve()