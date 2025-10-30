import sys
import threading

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    votes = list(map(int, data[2:]))

    # counts[i] will hold the number of votes candidate i has received
    counts = [0] * (N + 1)

    # current winner and the count of votes they have
    current_winner = 1
    max_count = 0

    out = []
    for v in votes:
        counts[v] += 1
        c = counts[v]
        # If this candidate now has more votes than the previous max, or
        # ties the max but has a smaller candidate number, update winner
        if c > max_count or (c == max_count and v < current_winner):
            current_winner = v
            max_count = c
        out.append(str(current_winner))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()