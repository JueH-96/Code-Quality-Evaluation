#!/usr/bin/env python3
import sys

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    L = int(data[0])
    R = int(data[1])
    res = []
    cur = L
    # Greedily at each step take the largest power-of-two block
    # that is aligned at cur and does not exceed R.
    while cur < R:
        diff = R - cur
        # lowbit = largest power-of-two dividing cur
        lowbit = cur & -cur
        # If cur==0, treat alignment as unlimited, i.e., up to diff
        if lowbit == 0:
            max_align = diff
        else:
            max_align = lowbit
        # We can use at most min(max_align, diff)
        cap = max_align if max_align <= diff else diff
        # Find the highest power-of-two <= cap
        # bit_length-1 gives floor(log2(cap))
        t = 1 << (cap.bit_length() - 1)
        res.append((cur, cur + t))
        cur += t

    # Output
    out = []
    out.append(str(len(res)))
    for l, r in res:
        out.append(f"{l} {r}")
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()