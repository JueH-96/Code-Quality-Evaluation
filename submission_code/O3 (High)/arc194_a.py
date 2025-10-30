import sys

def main() -> None:
    INF = 10 ** 18          # enough for constraints
    NEG = -INF

    it = sys.stdin.buffer.read().split()
    if not it:
        return
    N = int(it[0])
    A = list(map(int, it[1:]))

    best_odd  = NEG   # best sum of a valid sequence whose last chosen index is odd
    best_even = NEG   # best sum of a valid sequence whose last chosen index is even

    for idx, val in enumerate(A, 1):          # indices are 1-based
        if idx & 1:                           # current index is odd
            # candidate 1: start a new sequence with this single element
            cand = val
            # candidate 2: extend a sequence that currently ends at an even index
            if best_even != NEG:
                cand = max(cand, best_even + val)

            best_odd = max(best_odd, cand)
        else:                                 # current index is even
            # can only extend a sequence that currently ends at an odd index
            if best_odd != NEG:
                best_even = max(best_even, best_odd + val)

    if N & 1:             # N is odd -> final kept sequence must end at an odd index
        ans = best_odd
    else:                 # N is even -> either end at an even index or keep nothing
        ans = max(0, best_even)

    print(ans)

if __name__ == "__main__":
    main()