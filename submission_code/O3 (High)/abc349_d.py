import sys

# -------- auxiliary bit functions --------------------------------
def ctz(x: int) -> int:
    """Count trailing zero bits of a positive integer."""
    return (x & -x).bit_length() - 1     # x > 0

# -------- main ----------------------------------------------------
def main() -> None:
    L, R = map(int, sys.stdin.read().split())

    blocks = []
    cur = L
    while cur < R:
        remaining = R - cur

        k2 = remaining.bit_length() - 1               # floor log2(remaining)

        if cur == 0:
            k1 = 61                                   # "infinite" trailing zeros
        else:
            k1 = ctz(cur)

        k = min(k1, k2)
        length = 1 << k
        blocks.append((cur, cur + length))
        cur += length

    # -------- output ---------------------------------------------
    print(len(blocks))
    for l, r in blocks:
        print(l, r)

# ---------------------------------------------------------------
if __name__ == "__main__":
    main()