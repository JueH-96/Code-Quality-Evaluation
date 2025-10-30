import sys

def main() -> None:
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    counts = [0] * (N + 1)          # votes per candidate
    max_votes = 0                   # current highest vote count
    winner = 0                      # current winner's id (meaningless while max_votes == 0)

    out_lines = []
    for _ in range(M):
        cand = int(next(it))
        counts[cand] += 1
        c_votes = counts[cand]

        # update the current winner if necessary
        if c_votes > max_votes:
            max_votes = c_votes
            winner = cand
        elif c_votes == max_votes and cand < winner:
            winner = cand

        out_lines.append(str(winner))

    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()