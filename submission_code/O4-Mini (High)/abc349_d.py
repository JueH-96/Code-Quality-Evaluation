import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    L, R = map(int, data)
    cur = L
    segments = []
    while cur < R:
        rem = R - cur
        # largest power-of-two <= rem
        u = rem.bit_length() - 1
        # largest power-of-two dividing cur
        if cur == 0:
            # treat 0 as divisible by any 2^i up to needed range
            t = 60
        else:
            # (cur & -cur) is 2^v2(cur), its bit_length()-1 is v2(cur)
            t = (cur & -cur).bit_length() - 1
        # choose the smaller exponent so that 2^i divides cur and fits in rem
        i = t if t < u else u
        length = 1 << i
        segments.append((cur, cur + length))
        cur += length

    # output
    out = [str(len(segments))]
    for l, r in segments:
        out.append(f"{l} {r}")
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()