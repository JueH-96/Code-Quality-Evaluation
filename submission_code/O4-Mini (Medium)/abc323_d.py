import sys
import threading

def main():
    import sys

    data = sys.stdin
    N = int(data.readline())
    groups = {}  # maps odd-part B to dict of {k: count}

    for _ in range(N):
        line = data.readline().split()
        S = int(line[0]); C = int(line[1])
        # compute trailing zeros count k
        # S & -S = lowest set bit as power of two
        low = S & -S
        k = low.bit_length() - 1
        B = S >> k
        d = groups.get(B)
        if d is None:
            d = {}
            groups[B] = d
        d[k] = d.get(k, 0) + C

    ans = 0
    for B, cs in groups.items():
        # cs: dict mapping k -> count
        # total count in this group
        total_cnt = 0
        # also find min_k and max_k in this cs
        # cs is small (at most ~30 entries), so we can iterate
        min_k = None
        max_k = -1
        for k, c in cs.items():
            total_cnt += c
            if min_k is None or k < min_k: min_k = k
            if k > max_k: max_k = k

        # how many bits might carry propagate?
        # sum bit_length gives upper bound on extra levels
        extra = total_cnt.bit_length()
        # we'll iterate k from min_k up to at most max_k+extra
        carry = 0
        # iterate
        # we break early if beyond original max_k and carry is zero
        upper = max_k + extra
        for k in range(min_k, upper + 1):
            c = cs.get(k, 0)
            tot = c + carry
            # remainder at this level contributes to final slimes
            if tot & 1:
                ans += 1
            carry = tot >> 1
            # if we've passed original keys and no more carry, we can stop
            if k > max_k and carry == 0:
                break

    # output answer
    print(ans)

if __name__ == "__main__":
    main()