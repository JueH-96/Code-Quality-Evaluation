import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    # Parse N, M
    it = iter(data)
    try:
        N = int(next(it))
    except StopIteration:
        return
    M = int(next(it))
    # Initialize vote counts
    counts = [0] * (N + 1)
    max_votes = 0
    current_winner = N + 1  # so any real candidate will be smaller on tie

    out = []
    for _ in range(M):
        a = int(next(it))
        counts[a] += 1
        c = counts[a]
        # Update winner if this candidate leads alone, or ties but has smaller number
        if c > max_votes:
            max_votes = c
            current_winner = a
        elif c == max_votes and a < current_winner:
            current_winner = a
        out.append(str(current_winner))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()