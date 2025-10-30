import sys

def main() -> None:
    S = sys.stdin.readline().strip()
    n = len(S)

    # pos : position in S that we still have to match
    # idx : length of the alternating pattern already produced
    pos = 0
    idx = 0                  # 0-based index in the infinite pattern  i o i o …

    while pos < n:
        expected = 'i' if idx % 2 == 0 else 'o'   # character at position idx of the pattern
        if S[pos] == expected:                    # can embed this S character here
            pos += 1
        idx += 1                                  # move to next position of the pattern

    # idx  … minimal length of a pattern that contains S as a subsequence
    if idx % 2 == 1:                              # required length must be even
        idx += 1

    print(idx - n)                                # number of inserted characters

if __name__ == "__main__":
    main()