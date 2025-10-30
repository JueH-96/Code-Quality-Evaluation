import sys

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    n = data[0]
    P = data[1:1 + n]          # who each person is staring at
    Q = data[1 + n:1 + 2 * n]  # bib numbers each person is wearing

    # bib_to_person[b] = index (0-based) of the person whose bib number is b
    bib_to_person = [0] * (n + 1)
    for idx, bib in enumerate(Q):
        bib_to_person[bib] = idx

    S = [0] * n  # answer array, S[i] corresponds to bib number i+1
    for bib in range(1, n + 1):
        wearer_idx = bib_to_person[bib]       # person wearing bib `bib`
        target_idx = P[wearer_idx] - 1        # person they stare at (0-based)
        S[bib - 1] = Q[target_idx]            # bib number on that target person

    sys.stdout.write(' '.join(map(str, S)))

if __name__ == "__main__":
    main()