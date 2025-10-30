import sys

def main() -> None:
    S = sys.stdin.readline().strip()          # permutation of A..Z

    # position of each letter on the number line (1-indexed)
    pos = {c: i + 1 for i, c in enumerate(S)}

    total = 0
    curr = pos['A']                           # finger starts on 'A'
    for code in range(ord('B'), ord('Z') + 1):
        nxt = pos[chr(code)]
        total += abs(nxt - curr)              # move directly to next letter
        curr = nxt

    print(total)

if __name__ == "__main__":
    main()