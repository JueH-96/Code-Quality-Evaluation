import sys

def tile_k(x: int, y: int) -> int:
    """
    Return the horizontal index k of the tile that contains the unit square (x,y).
    For a row with parity p = y&1 the left border of the tile is
    L = 2*k + p, so
        k = floor((x - p)/2)
    """
    return (x - (y & 1)) // 2


def main() -> None:
    sx, sy, tx, ty = map(int, sys.stdin.read().split())

    # tile indices of the start / goal positions
    ks = tile_k(sx, sy)
    kt = tile_k(tx, ty)

    dk = kt - ks                 # horizontal difference in the tile graph
    dy = ty - sy                 # vertical difference of rows
    D  = abs(dy)                 # number of vertical moves if we never turn back

    # how many vertical steps of the D start from even / odd rows
    if D % 2 == 0:
        even_leaves = odd_leaves = D // 2
    else:
        if sy & 1:                         # start row is odd
            odd_leaves  = D // 2 + 1
            even_leaves = D // 2
        else:                              # start row is even
            odd_leaves  = D // 2
            even_leaves = D // 2 + 1

    # vertical steps can raise k at most `odd_leaves` and lower it at most `even_leaves`
    if dk >= 0:
        extra = max(0, dk - odd_leaves)          # still have to increase k to the right
    else:
        extra = max(0, -dk - even_leaves)        # still have to decrease k to the left

    # each remaining unit of |dk| needs one horizontal boundary crossing
    ans = D + extra
    print(ans)


if __name__ == "__main__":
    main()