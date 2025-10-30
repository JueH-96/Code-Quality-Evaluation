import sys

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    N = data[0]
    A = data[1:]          # already sorted in non-decreasing order

    half = N // 2         # at most this many pairs are possible
    small = 0             # pointer to the current “small” candidate
    large = half          # pointer to the current “large” candidate
    pairs = 0

    # try to make a pair while both pointers are in range
    while small < half and large < N:
        if A[small] * 2 <= A[large]:
            # we can stack A[small] on A[large]
            pairs += 1
            small += 1
            large += 1
        else:
            # A[large] is not big enough, look for a larger mochi
            large += 1

    print(pairs)

if __name__ == "__main__":
    main()