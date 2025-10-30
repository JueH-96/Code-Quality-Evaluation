import sys
from collections import defaultdict

def main() -> None:
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    N = int(next(it))
    T = int(next(it))

    # Current score of each player (1-indexed)
    scores = [0] * (N + 1)

    # counts[v] = number of players whose score is exactly v
    counts = {0: N}
    distinct = 1          # how many different score values exist now

    out_lines = []

    for _ in range(T):
        a = int(next(it))
        b = int(next(it))

        old = scores[a]

        # remove the old score of player a from the multiset
        old_cnt = counts[old]
        if old_cnt == 1:
            del counts[old]
            distinct -= 1
        else:
            counts[old] = old_cnt - 1

        # add the new score
        new = old + b
        scores[a] = new
        if new in counts:
            counts[new] += 1
        else:
            counts[new] = 1
            distinct += 1

        out_lines.append(str(distinct))

    sys.stdout.write('
'.join(out_lines))

if __name__ == "__main__":
    main()