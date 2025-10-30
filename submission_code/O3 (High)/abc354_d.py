import sys

def count_by_mod4(start: int, length: int):
    """
    return a list cnt[0..3] where cnt[r] is the number of integers n
    such that start <= n < start+length and n % 4 == r
    """
    base = length // 4
    cnt = [base, base, base, base]
    extra = length % 4
    first = start % 4
    for i in range(extra):
        cnt[(first + i) % 4] += 1
    return cnt


def main() -> None:
    A, B, C, D = map(int, sys.stdin.readline().split())

    width = C - A          # number of vertical stripes
    height = D - B         # total height

    # counts of stripes for each remainder modulo 4
    k = count_by_mod4(A, width)   # k[0] .. k[3]

    # constants that depend only on the remainder of stripe index
    # d  : full 2-height cell difference (+1,+1,-1,-1)
    # bs : difference of the TOP half of a cell (present when bottom y is odd)
    # ts : difference of the BOTTOM half of a cell (present when top   y is odd)
    d  = [ 1,  1, -1, -1]
    bs = [ 0,  1,  0, -1]          # d - g(1)
    ts = [ 1,  0, -1,  0]          # g(1)

    # pre-computed sums using the counts
    sum_d  = k[0] + k[1] - k[2] - k[3]
    sum_bs = k[1] - k[3]           # k1 - k3
    sum_ts = k[0] - k[2]           # k0 - k2

    # number of complete 2-height layers fully contained
    J = height // 2

    # is bottom / top y coordinate odd?
    B_odd = B & 1
    D_odd = D & 1

    # integral of s(x,y) (= black – white area) inside the rectangle
    S = J * sum_d
    if B_odd:
        S += sum_bs
    if D_odd:
        S += sum_ts

    # rectangle area
    area = width * height

    # required value: 2 * (black area) = area + S
    ans = area + S
    print(ans)


if __name__ == "__main__":
    main()