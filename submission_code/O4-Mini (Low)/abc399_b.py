def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    P = list(map(int, data[1:]))

    # Create a sorted list of scores in descending order
    sorted_scores = sorted(P, reverse=True)

    # Build a mapping from score -> competition rank
    rank_map = {}
    for i, score in enumerate(sorted_scores):
        # Only assign the rank when we first see this score
        if score not in rank_map:
            # In competition ranking, the rank is the 1-based index
            # of the first occurrence of this score in the sorted list
            rank_map[score] = i + 1

    # Output the rank for each original participant
    out = []
    for score in P:
        out.append(str(rank_map[score]))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()