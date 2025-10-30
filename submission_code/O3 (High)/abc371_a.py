def main() -> None:
    import sys

    # Read the three inequality symbols
    data = sys.stdin.read().strip().split()
    if len(data) != 3:           # safety, though judge will always give 3 tokens
        return
    S_AB, S_AC, S_BC = data

    # Score = number of brothers someone is older than
    score = {'A': 0, 'B': 0, 'C': 0}

    # A vs B
    if S_AB == '<':      # A younger, so B older
        score['B'] += 1
    else:                # A older
        score['A'] += 1

    # A vs C
    if S_AC == '<':
        score['C'] += 1
    else:
        score['A'] += 1

    # B vs C
    if S_BC == '<':
        score['C'] += 1
    else:
        score['B'] += 1

    # The middle brother is older than exactly one brother → score == 1
    for name, sc in score.items():
        if sc == 1:
            print(name)
            break


if __name__ == "__main__":
    main()