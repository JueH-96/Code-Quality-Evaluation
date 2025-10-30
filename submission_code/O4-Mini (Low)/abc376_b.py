import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    N = int(data[0])
    Q = int(data[1])
    idx = 2

    # Compute the minimal number of steps on an N‐node cycle from a to b,
    # without passing through 'blocked'. Guaranteed that b != blocked.
    def dist(a, b, blocked):
        if a == b:
            return 0
        # clockwise distance
        cw = (b - a) % N
        # check if blocked lies on the cw path from a to b
        ok_cw = True
        # we step i=1..cw, node = (a + i) mod N (1-based)
        for i in range(1, cw + 1):
            node = ((a - 1 + i) % N) + 1
            if node == blocked:
                ok_cw = False
                break
        # counter‐clockwise distance
        ccw = (a - b) % N
        ok_ccw = True
        for i in range(1, ccw + 1):
            node = ((a - 1 - i) % N) + 1
            if node == blocked:
                ok_ccw = False
                break
        res = []
        if ok_cw:
            res.append(cw)
        if ok_ccw:
            res.append(ccw)
        # one of them must be valid as per problem guarantee
        return min(res)

    # initial positions
    lpos = 1
    rpos = 2
    total = 0

    for _ in range(Q):
        H = data[idx]; idx += 1
        T = int(data[idx]); idx += 1
        if H == 'L':
            # move left from lpos -> T, avoiding rpos
            d = dist(lpos, T, rpos)
            total += d
            lpos = T
        else:
            # move right from rpos -> T, avoiding lpos
            d = dist(rpos, T, lpos)
            total += d
            rpos = T

    print(total)

if __name__ == "__main__":
    main()