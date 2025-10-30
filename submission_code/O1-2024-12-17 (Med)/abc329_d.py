def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    votes = list(map(int, data[2:]))

    counts = [0] * (N+1)  # 1-based index
    max_count = 0
    current_winner = 1

    # We'll collect outputs and print later (to minimize I/O overhead)
    results = []

    for v in votes:
        counts[v] += 1
        if counts[v] > max_count:
            max_count = counts[v]
            current_winner = v
        elif counts[v] == max_count and v < current_winner:
            current_winner = v
        results.append(str(current_winner))

    print("
".join(results))

# Do not forget to call main()
main()