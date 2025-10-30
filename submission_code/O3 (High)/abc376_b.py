import sys

def distance(s: int, b: int, t: int, N: int) -> int:
    """
    minimal number of steps to move from s to t
    without stepping on the forbidden part b
    """
    if s == t:                             # already there
        return 0

    cw = (t - s) % N                       # clockwise distance (1 … N-1)
    ccw = (s - t) % N                      # counter-clockwise distance
    diff_b = (b - s) % N                   # distance from s to b clockwise

    # is the clockwise path blocked?
    if 0 < diff_b <= cw:                   # b lies on clockwise path
        return ccw
    else:
        return cw

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    Q = int(next(it))

    left = 1                               # initial positions
    right = 2
    total = 0

    for _ in range(Q):
        hand = next(it)
        target = int(next(it))

        if hand == 'L':                    # move left hand
            total += distance(left, right, target, N)
            left = target                  # update position
        else:                              # move right hand
            total += distance(right, left, target, N)
            right = target

    print(total)

if __name__ == "__main__":
    main()