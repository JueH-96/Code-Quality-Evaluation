import sys

def main() -> None:
    """
    For every slime size S we split it into
        S = odd_part * 2^k     (odd_part is odd, k >= 0)

    Inside the group having the same odd_part, slimes can move between
    the “levels” (k) exactly like carrying in binary addition.
    If for this odd_part the numbers of slimes on each level are
    (c_0, c_1, c_2, …), then the whole situation is equivalent to the
    binary integer

        T = Σ  c_k · 2^k

    After performing all possible syntheses this integer is reduced to
    its standard binary form and the amount of slimes becomes the number
    of 1-bits in T (its popcount).  
    Different odd_part groups never interact, so the final answer is the
    sum of these popcounts over all groups.

    Complexity
    ----------
    • O(N) time, one pass over the input.  
    • O(M) memory, where M ≤ number of different odd parts (≤ N).
    """
    input = sys.stdin.readline
    N = int(input().strip())

    totals = {}                       # key: odd_part, value: T described above

    for _ in range(N):
        S, C = map(int, input().split())

        # number of trailing zeros of S
        low = S & -S                  # lowest set bit (a power of two)
        k = low.bit_length() - 1      # exponent of this power of two

        odd = S >> k                  # remove all factors of two
        totals[odd] = totals.get(odd, 0) + (C << k)

    # Python 3.8+ has int.bit_count(); provide a fallback just in case
    if hasattr(int, 'bit_count'):
        popcount = int.bit_count
    else:
        popcount = lambda x: bin(x).count('1')

    answer = sum(popcount(v) for v in totals.values())
    print(answer)

if __name__ == "__main__":
    main()