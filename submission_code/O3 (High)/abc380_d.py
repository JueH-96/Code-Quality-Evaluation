import sys
import bisect

def main() -> None:
    # fast input
    data = sys.stdin.buffer.read().split()
    S = data[0].decode()                   # original string
    n = len(S)
    Q = int(data[1])
    Ks = list(map(int, data[2:2 + Q]))

    max_K = max(Ks)

    # pre–compute the successive lengths  n, n*2, n*4, …  up to >= max_K
    lengths = [n]
    while lengths[-1] < max_K:
        lengths.append(lengths[-1] << 1)   # multiply by 2

    # helper to toggle one character
    def toggle(c: str) -> str:
        # swap case for English letters
        return c.lower() if c.isupper() else c.upper()

    res = []
    for K in Ks:
        # smallest length that is at least K
        len_cur = lengths[bisect.bisect_left(lengths, K)]
        parity = 0                         # 0: as is, 1: toggled an odd number of times

        # walk down the doubling tree
        while len_cur > n:
            half = len_cur >> 1
            if K > half:                   # in the right (toggled) half
                K -= half
                parity ^= 1
            len_cur = half

        ch = S[K - 1]
        if parity:
            ch = toggle(ch)
        res.append(ch)

    sys.stdout.write(' '.join(res))

if __name__ == "__main__":
    main()